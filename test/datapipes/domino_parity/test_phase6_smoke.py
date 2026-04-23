# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Phase 6 smoke test: synthetic sample end-to-end through DoMINO.

Minimal proof that the fold-in is complete enough to train.  This is NOT
the byte-parity harness (``test_e2e_parity.py``); it is a coarse sanity
check that:

1. The recipe-local transforms import and register as ``${dp:...}`` names.
2. ``DomainMeshReader`` + ``MeshDataset`` can load the synthetic fixture
   without crashing.
3. Every key ``DoMINO.forward`` requires for ``model_type="surface"`` is
   present in the output dict with the right rank / dtype.
4. ``DoMINO.forward`` runs on one batch and returns surface output of the
   expected shape.

Runs in seconds on a single GPU.  Skipped automatically on login nodes
(no warp / cuml / GPU).

Execution
---------

.. code-block:: bash

   # Inside container, on a compute node with 1 GPU
   pytest test/datapipes/domino_parity/test_phase6_smoke.py -v -s
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


def test_phase6_smoke_surface(tmp_path: Path) -> None:
    """End-to-end smoke test for the surface HLPW-DoMINO recipe."""
    pytest.importorskip("warp")
    pytest.importorskip("tensordict")

    if not torch.cuda.is_available():
        pytest.skip("Phase-6 smoke needs a GPU for DoMINO.forward")

    # ---- Wire up the recipe-local transforms. ----
    import sys

    recipe_src = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "cfd"
        / "external_aerodynamics"
        / "unified_external_aero_recipe"
        / "src"
    )
    if str(recipe_src) not in sys.path:
        sys.path.insert(0, str(recipe_src))

    import domino_transforms  # noqa: F401  registers ${dp:...} names
    import sdf  # noqa: F401

    from physicsnemo.datapipes.mesh_dataset import MeshDataset
    from physicsnemo.datapipes.readers.mesh import DomainMeshReader
    from physicsnemo.datapipes.transforms.mesh import (
        CenterMesh,
        ComputeSurfaceNormals,
        MeshToTensorDict,
        SubsampleMesh,
    )
    from physicsnemo.models.domino import DoMINO

    from domino_transforms import (
        ComputeDoMINOPositionalEncodings,
        ComputeGridSDFFromBoundary,
        SurfaceKNNNeighbors,
    )
    from sdf import ComputeSDFFromBoundary, DropBoundary

    from .fixtures.make_synthetic_sample import write_synthetic_sample

    # ---- Write synthetic sample. ----
    paths = write_synthetic_sample(
        tmp_path, seed=42, n_interior=2000, surface_subdivisions=2, case_name="case_000"
    )

    # ---- Build the transform chain (subset of hlpw_domino_surface.yaml). ----
    sampling_resolution = 128
    transforms = [
        CenterMesh(use_area_weighting=True),
        ComputeGridSDFFromBoundary(
            boundary_name="stl_geometry",
            grid_resolution=(8, 8, 8),
            grid_field="surf_grid",
            sdf_field="sdf_surf_grid",
        ),
        ComputeSurfaceNormals(store_as="cell_data", field_name="normals"),
        SubsampleMesh(n_cells=sampling_resolution),
        SurfaceKNNNeighbors(
            k=4,
            boundary_name="boundary",
            neighbors_field="surface_mesh_neighbors",
            neighbor_normals_field="surface_neighbors_normals",
            neighbor_areas_field="surface_neighbors_areas",
        ),
        ComputeDoMINOPositionalEncodings(
            stl_boundary="stl_geometry",
            target_boundary="boundary",
            compute_volume_encodings=False,
            compute_surface_encodings=True,
        ),
        DropBoundary(names=["stl_geometry"]),
        MeshToTensorDict(),
    ]

    reader = DomainMeshReader(
        path=tmp_path,
        pattern="**/*.pdmsh",
        extra_boundaries={
            "stl_geometry": {"pattern": "*_single_solid.stl.pmsh"},
        },
    )
    dataset = MeshDataset(reader=reader, transforms=transforms, device="cuda")

    td, _meta = dataset[0]

    # ---- Assemble the surface-mode data_dict that DoMINO.forward expects. ----
    device = torch.device("cuda")
    bnd = td["boundaries"]["boundary"]
    cell_centroids = bnd["points"][bnd["cells"]].mean(dim=1).float()
    # Triangle areas from cells (to feed into surface_areas).
    tri = bnd["points"][bnd["cells"]].float()
    areas = 0.5 * torch.linalg.cross(
        tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1
    ).norm(dim=-1)

    batch = {
        "geometry_coordinates": bnd["points"].unsqueeze(0).float().to(device),
        "surf_grid": td["global_data"]["surf_grid"].unsqueeze(0).to(device),
        "sdf_surf_grid": td["global_data"]["sdf_surf_grid"].unsqueeze(0).to(device),
        "global_params_values": torch.zeros(1, 1, 1, dtype=torch.float32, device=device),
        "global_params_reference": torch.ones(1, 1, 1, dtype=torch.float32, device=device),
        "pos_surface_center_of_mass": bnd["cell_data"]["pos_surface_center_of_mass"]
        .unsqueeze(0)
        .float()
        .to(device),
        "surface_mesh_centers": cell_centroids.unsqueeze(0).to(device),
        "surface_mesh_neighbors": bnd["cell_data"]["surface_mesh_neighbors"]
        .unsqueeze(0)
        .float()
        .to(device),
        "surface_normals": bnd["cell_data"]["normals"].unsqueeze(0).float().to(device),
        "surface_neighbors_normals": bnd["cell_data"]["surface_neighbors_normals"]
        .unsqueeze(0)
        .float()
        .to(device),
        "surface_areas": areas.unsqueeze(0).to(device),
        "surface_neighbors_areas": bnd["cell_data"]["surface_neighbors_areas"]
        .unsqueeze(0)
        .float()
        .to(device),
    }

    # ---- Instantiate a tiny DoMINO model. ----
    from omegaconf import OmegaConf

    model_parameters = OmegaConf.create(
        {
            "interp_res": [8, 8, 8],
            "use_sdf_in_basis_func": True,
            "surface_neighbors": True,
            "num_neighbors_surface": 3,
            "use_surface_normals": True,
            "use_surface_area": True,
            "geometry_encoding_type": "both",
        }
    )

    model = DoMINO(
        input_features=3,
        output_features_vol=None,
        output_features_surf=3,  # synthetic sample has 3-channel surface output
        global_features=1,
        model_parameters=model_parameters,
    ).to(device)

    out_vol, out_surf = model(batch)
    assert out_vol is None, "volume output should be None in surface-only mode"
    assert out_surf is not None
    # Expected shape: (batch=1, n_surface=sampling_resolution, out_features=3)
    assert out_surf.shape[0] == 1
    assert out_surf.shape[2] == 3
    assert torch.isfinite(out_surf).all(), "non-finite values in forward output"
