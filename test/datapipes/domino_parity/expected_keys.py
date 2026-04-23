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
Machine-checked spec of the DoMINO ``forward()`` input-dict contract.

Every key that :meth:`physicsnemo.models.domino.model.DoMINO.forward` reads from
``data_dict`` is listed here with its expected rank, dtype, and the transform
(or legacy pipeline step) that is responsible for producing it in the new
Mesh-native recipe.

The parity harness ``test_e2e_parity.py`` uses this spec to validate that

1. the new ``MeshDataset`` chain output contains **exactly** these keys (no
   more, no fewer), and
2. each tensor matches the corresponding legacy ``DoMINODataPipe`` output
   bit-for-bit under sampling-disabled settings.

The ``rank`` column below is **per-sample** -- it excludes the leading
batch dimension.  The new ``MeshDataset`` pipeline emits per-sample
tensors and relies on ``DataLoader`` collate to stack into ``(B, ...)``;
the legacy ``DoMINODataPipe`` prepends ``unsqueeze(0)`` itself at
``domino_datapipe.py:1007``, so the parity harness squeezes the legacy
side before comparing.

Missing-owner rows fail loudly at import time so that a new key can never be
added to DoMINO without updating this spec.

References
----------
DoMINO.forward() required keys are enumerated at
``physicsnemo/models/domino/model.py`` lines 522-555.
Legacy producer side is ``physicsnemo/datapipes/cae/domino_datapipe.py``
``DoMINODataPipe.process_data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class KeySpec:
    """Contract for a single DoMINO forward-input key.

    Parameters
    ----------
    name : str
        Key name as used in ``data_dict``.
    rank : int
        Expected number of tensor dimensions (before the dataloader's batch
        dim is prepended).  ``0`` means scalar.
    dtype : torch.dtype
        Expected element type.  For parity we only accept float32 for
        floating-point tensors; ints vary.
    owner : str
        Short name of the transform (or legacy step) that emits this key.
        Used by the harness to localise failures.
    mode : {"volume", "surface", "both"}
        Whether the key is required in ``model_type`` = volume, surface, or
        both.  Kept here (not on the transform) because the model's
        ``forward`` branches on the same flag.
    tolerance : dict[str, float] | None
        Optional per-key override of default ``torch.testing.assert_close``
        tolerances.  ``None`` means use the harness default.  BVH-dependent
        outputs (SDF grid, closest-point encodings) may need slightly looser
        ``atol``.
    """

    name: str
    rank: int
    dtype: torch.dtype
    owner: str
    mode: Literal["volume", "surface", "both"]
    tolerance: dict[str, float] | None = None


# Default per-dtype tolerances for torch.testing.assert_close.
DEFAULT_FLOAT32_RTOL = 1e-5
DEFAULT_FLOAT32_ATOL = 1e-6
# Looser ATOL for outputs that depend on Warp BVH geometry kernels where
# cross-run reproducibility is dominated by floating-point summation order.
BVH_FLOAT32_ATOL = 1e-4


EXPECTED_KEYS: tuple[KeySpec, ...] = (
    # ---- Always required (both volume and surface model paths) ----
    KeySpec(
        name="geometry_coordinates",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from stl_geometry boundary.points)",
        mode="both",
    ),
    KeySpec(
        name="surf_grid",
        rank=4,
        dtype=torch.float32,
        owner="ComputeGridSDFFromBoundary (surface-bbox grid)",
        mode="both",
    ),
    KeySpec(
        name="sdf_surf_grid",
        rank=3,
        dtype=torch.float32,
        owner="ComputeGridSDFFromBoundary (SDF on surface-bbox grid)",
        mode="both",
        tolerance={"atol": BVH_FLOAT32_ATOL, "rtol": DEFAULT_FLOAT32_RTOL},
    ),
    KeySpec(
        name="global_params_values",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from global_data)",
        mode="both",
    ),
    KeySpec(
        name="global_params_reference",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from global_data)",
        mode="both",
    ),
    # ---- Volume-only keys ----
    KeySpec(
        name="grid",
        rank=4,
        dtype=torch.float32,
        owner="ComputeGridSDFFromBoundary (volume-bbox grid)",
        mode="volume",
    ),
    KeySpec(
        name="sdf_grid",
        rank=3,
        dtype=torch.float32,
        owner="ComputeGridSDFFromBoundary (SDF on volume-bbox grid)",
        mode="volume",
        tolerance={"atol": BVH_FLOAT32_ATOL, "rtol": DEFAULT_FLOAT32_RTOL},
    ),
    KeySpec(
        name="sdf_nodes",
        rank=2,
        dtype=torch.float32,
        owner="ComputeSDFFromBoundary (library, point-level SDF)",
        mode="volume",
        tolerance={"atol": BVH_FLOAT32_ATOL, "rtol": DEFAULT_FLOAT32_RTOL},
    ),
    KeySpec(
        name="pos_volume_closest",
        rank=2,
        dtype=torch.float32,
        owner="ComputeDoMINOPositionalEncodings",
        mode="volume",
        tolerance={"atol": BVH_FLOAT32_ATOL, "rtol": DEFAULT_FLOAT32_RTOL},
    ),
    KeySpec(
        name="pos_volume_center_of_mass",
        rank=2,
        dtype=torch.float32,
        owner="ComputeDoMINOPositionalEncodings",
        mode="volume",
    ),
    KeySpec(
        name="volume_mesh_centers",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from interior.points)",
        mode="volume",
    ),
    KeySpec(
        name="volume_fields",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from interior.point_data fields)",
        mode="volume",
    ),
    # ---- Surface-only keys ----
    KeySpec(
        name="pos_surface_center_of_mass",
        rank=2,
        dtype=torch.float32,
        owner="ComputeDoMINOPositionalEncodings",
        mode="surface",
    ),
    KeySpec(
        name="surface_mesh_centers",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from boundary.cell_centroids)",
        mode="surface",
    ),
    KeySpec(
        name="surface_mesh_neighbors",
        rank=3,
        dtype=torch.float32,
        owner="SurfaceKNNNeighbors",
        mode="surface",
    ),
    KeySpec(
        name="surface_normals",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from boundary.cell_normals)",
        mode="surface",
    ),
    KeySpec(
        name="surface_neighbors_normals",
        rank=3,
        dtype=torch.float32,
        owner="SurfaceKNNNeighbors",
        mode="surface",
    ),
    KeySpec(
        name="surface_areas",
        rank=1,
        dtype=torch.float32,
        owner="RestructureTensorDict (from boundary.cell_areas)",
        mode="surface",
    ),
    KeySpec(
        name="surface_neighbors_areas",
        rank=2,
        dtype=torch.float32,
        owner="SurfaceKNNNeighbors",
        mode="surface",
    ),
    KeySpec(
        name="surface_fields",
        rank=2,
        dtype=torch.float32,
        owner="RestructureTensorDict (from boundary.cell_data fields)",
        mode="surface",
    ),
)


def keys_for_mode(mode: Literal["volume", "surface", "combined"]) -> set[str]:
    """Return the set of required ``data_dict`` keys for a given DoMINO mode.

    Parameters
    ----------
    mode : {"volume", "surface", "combined"}
        DoMINO ``model_type``. ``"combined"`` requires the union of volume and
        surface keys.

    Returns
    -------
    set[str]
        Key names the harness must find in the pipeline output dict.
    """
    if mode == "volume":
        wanted = {"both", "volume"}
    elif mode == "surface":
        wanted = {"both", "surface"}
    elif mode == "combined":
        wanted = {"both", "volume", "surface"}
    else:
        raise ValueError(f"Unknown DoMINO model_type: {mode!r}")
    return {spec.name for spec in EXPECTED_KEYS if spec.mode in wanted}


def spec_by_name() -> dict[str, KeySpec]:
    """Return a dict mapping key name to :class:`KeySpec` for lookup."""
    return {spec.name: spec for spec in EXPECTED_KEYS}


# Fail loud at import time if the spec is ever malformed.  Each required
# DoMINO key must appear exactly once and have a non-empty owner string.
def _validate() -> None:
    names = [s.name for s in EXPECTED_KEYS]
    if len(set(names)) != len(names):
        duplicates = sorted({n for n in names if names.count(n) > 1})
        raise RuntimeError(f"EXPECTED_KEYS contains duplicates: {duplicates}")
    for s in EXPECTED_KEYS:
        if not s.owner:
            raise RuntimeError(f"KeySpec {s.name!r} has empty owner")
        if s.mode not in ("volume", "surface", "both"):
            raise RuntimeError(f"KeySpec {s.name!r} has bad mode {s.mode!r}")


_validate()
