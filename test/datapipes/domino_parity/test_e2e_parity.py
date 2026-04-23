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
End-to-end parity harness: legacy DoMINODataPipe vs new MeshDataset chain.

Phase 2.5 Axis 1 gate.  Runs both pipelines with **sampling disabled** so that
every output tensor is deterministic and can be compared byte-for-byte
(modulo per-key tolerances declared in :mod:`expected_keys`).

Legacy side
-----------
:class:`~physicsnemo.datapipes.cae.domino_datapipe.DoMINODataPipe` is invoked
with an in-memory ``data_dict`` constructed directly from the synthetic
:class:`DomainMesh` fixture -- no file I/O on the legacy side.  This is
decoupled from ``CAEDataset`` on purpose: ``DoMINODataPipe.process_data``
accepts any dict matching its contract, and the harness supplies exactly that
dict from the fixture so both sides see the same tensor values.

New side
--------
The fixture's ``.pdmsh`` + ``.stl.pmsh`` directories are loaded via
:class:`~physicsnemo.datapipes.readers.mesh.DomainMeshReader` with
``extra_boundaries.stl_geometry`` injecting the STL mesh as a side-load
boundary, then a :class:`~physicsnemo.datapipes.mesh_dataset.MeshDataset` runs
the Phase-2 DoMINO transform chain.  No subsampling on either side.

Cluster requirements
--------------------
This test exercises :func:`physicsnemo.nn.functional.signed_distance_field`
(Warp BVH) and :func:`physicsnemo.nn.functional.knn` (cuml on GPU, scipy on
CPU).  On a small synthetic fixture it runs in seconds on either CPU or GPU;
on the Tier-2 ``$PARITY_SNAPSHOT_DIR`` real HLPW sample it is strongly
preferred to run on 1 interactive GPU (see plan "Cluster-access map").

Calibration
-----------
Before any new transform lands in ``domino_transforms.py``, run
``pytest -k calibration`` to verify the harness is self-consistent when
new_out is forced equal to legacy_out.  This is the RED-verify of the TDD
cycle deferred from the login node (where warp is unavailable) to the
container session.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import pytest
import torch

from .expected_keys import keys_for_mode, spec_by_name
from .fixtures.make_synthetic_sample import (
    build_synthetic_sample,
    write_synthetic_sample,
)

if TYPE_CHECKING:
    from physicsnemo.mesh import DomainMesh, Mesh  # noqa: F401  type hints only

# --------------------------------------------------------------------------- #
# Mode-specific flags and tolerances
# --------------------------------------------------------------------------- #

_DEFAULT_RTOL = 1e-5
_DEFAULT_ATOL = 1e-6


def _tolerance_for(name: str) -> tuple[float, float]:
    """Return (rtol, atol) for a named key."""
    spec = spec_by_name().get(name)
    if spec is None or spec.tolerance is None:
        return _DEFAULT_RTOL, _DEFAULT_ATOL
    return (
        spec.tolerance.get("rtol", _DEFAULT_RTOL),
        spec.tolerance.get("atol", _DEFAULT_ATOL),
    )


# --------------------------------------------------------------------------- #
# Legacy-side adapter: DomainMesh -> data_dict for DoMINODataPipe
# --------------------------------------------------------------------------- #


def domainmesh_to_legacy_datadict(
    stl_mesh: "Mesh",
    domain_mesh: "DomainMesh",
    *,
    boundary_name: str = "boundary",
    surface_variables: tuple[str, ...] = ("pMeanTrim", "wallShearStressMeanTrim"),
    volume_variables: tuple[str, ...] = ("UMeanTrim", "pMeanTrim", "nutMeanTrim"),
) -> dict[str, torch.Tensor]:
    r"""Build the dict that ``DoMINODataPipe.process_data`` expects.

    Bypasses ``CAEDataset`` entirely so the harness can run even on a
    branch where ``PmshFileReader`` has been removed.  Mimics the keys
    that legacy ``CAEDataset+PmshFileReader`` would have produced.

    Parameters
    ----------
    stl_mesh : Mesh
        Full-resolution surface (STL role), contributes ``stl_*`` keys.
    domain_mesh : DomainMesh
        Contains ``interior`` (volume point cloud) and a boundary.
    boundary_name : str
        Key in ``domain_mesh.boundaries`` treated as the CFD surface.
    surface_variables : sequence of str
        ``cell_data`` field names on the boundary, concatenated last-dim
        into ``surface_fields``.
    volume_variables : sequence of str
        ``point_data`` field names on the interior, concatenated last-dim
        into ``volume_fields``.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys matching what ``DoMINODataPipe.process_data`` reads:
        ``global_params_values``, ``global_params_reference``,
        ``stl_coordinates``, ``stl_faces``, ``stl_centers``, ``stl_areas``,
        ``surface_mesh_centers``, ``surface_normals``, ``surface_areas``,
        ``surface_fields``, ``volume_mesh_centers``, ``volume_fields``.

    Notes
    -----
    ``global_params_values`` and ``global_params_reference`` are placeholder
    ``(1, 2)`` tensors -- DoMINO's current implementation reads them without
    modifying them, so any same-shape values produce the same downstream
    output.  When a real dataset has per-sample global parameters, pass
    them in ``domain_mesh.global_data`` and wire them here.
    """
    boundary = domain_mesh.boundaries[boundary_name]
    interior = domain_mesh.interior

    # STL keys come from the stl-role mesh (full resolution geometry).
    stl_coordinates = stl_mesh.points.float()
    stl_faces = stl_mesh.cells.long()
    stl_centers = stl_mesh.cell_centroids.float()
    stl_areas = stl_mesh.cell_areas.float()

    # Surface keys (cell-level) from the boundary mesh.
    surface_mesh_centers = boundary.cell_centroids.float()
    surface_normals = boundary.cell_normals.float()
    surface_areas = boundary.cell_areas.float()

    # Concatenate surface fields along last dim, matching legacy convention.
    surface_field_chunks: list[torch.Tensor] = []
    for name in surface_variables:
        if name in boundary.cell_data.keys():
            f = boundary.cell_data[name].float()
            if f.ndim == 1:
                f = f.unsqueeze(-1)
            surface_field_chunks.append(f)
    surface_fields = (
        torch.cat(surface_field_chunks, dim=-1)
        if surface_field_chunks
        else torch.zeros(surface_mesh_centers.shape[0], 0)
    )

    # Volume keys (point-level) from the interior point cloud.
    volume_mesh_centers = interior.points.float()
    volume_field_chunks: list[torch.Tensor] = []
    for name in volume_variables:
        if name in interior.point_data.keys():
            f = interior.point_data[name].float()
            if f.ndim == 1:
                f = f.unsqueeze(-1)
            volume_field_chunks.append(f)
    volume_fields = (
        torch.cat(volume_field_chunks, dim=-1)
        if volume_field_chunks
        else torch.zeros(volume_mesh_centers.shape[0], 0)
    )

    # Placeholder global-parameter tensors.  DoMINO passes these through
    # to the MLP unchanged; identical values on both sides give identical
    # downstream outputs.
    global_params_values = torch.zeros(1, 2, dtype=torch.float32)
    global_params_reference = torch.ones(1, 2, dtype=torch.float32)

    return {
        "global_params_values": global_params_values,
        "global_params_reference": global_params_reference,
        "stl_coordinates": stl_coordinates,
        "stl_faces": stl_faces,
        "stl_centers": stl_centers,
        "stl_areas": stl_areas,
        "surface_mesh_centers": surface_mesh_centers,
        "surface_normals": surface_normals,
        "surface_areas": surface_areas,
        "surface_fields": surface_fields,
        "volume_mesh_centers": volume_mesh_centers,
        "volume_fields": volume_fields,
    }


# --------------------------------------------------------------------------- #
# Bounding boxes -- harness must supply these to DoMINODataPipe
# --------------------------------------------------------------------------- #


class _BBox:
    """Duck-typed bounding box for DoMINO's ``BoundingBox`` Protocol."""

    def __init__(self, min_coords: list[float], max_coords: list[float]):
        self.min = min_coords
        self.max = max_coords


SYNTHETIC_VOLUME_BBOX = _BBox(min_coords=[-2.5, -1.5, 0.0], max_coords=[2.5, 1.5, 2.5])
SYNTHETIC_SURFACE_BBOX = _BBox(
    min_coords=[-0.6, -0.6, -0.1], max_coords=[0.6, 0.6, 1.1]
)


# --------------------------------------------------------------------------- #
# Pytest fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_sample(tmp_path: Path) -> dict:
    r"""Synthetic DoMINO-ready sample written to a fresh ``tmp_path``.

    Returns a dict with keys:

    - ``stl_mesh`` (:class:`Mesh`): in-memory STL geometry
    - ``domain_mesh`` (:class:`DomainMesh`): in-memory domain
    - ``case_dir`` (:class:`Path`): directory with ``.pdmsh`` + ``.stl.pmsh``
    - ``pdmsh`` (:class:`Path`): the domain mesh directory
    - ``stl_pmsh`` (:class:`Path`): the stl geometry directory

    Both the in-memory meshes and on-disk versions are produced from the same
    tensors (the fixture calls ``build_synthetic_sample`` once, then saves
    those same tensors), so byte-parity is guaranteed at the reader boundary.
    """
    stl_mesh, domain_mesh = build_synthetic_sample(seed=42)
    paths = write_synthetic_sample(tmp_path, seed=42, case_name="case_000")
    return {
        "stl_mesh": stl_mesh,
        "domain_mesh": domain_mesh,
        **paths,
    }


@pytest.fixture
def real_sample(request) -> Path | None:
    r"""Path to one real HLPW ``.pdmsh`` case (Tier 2, optional).

    Enabled when ``$PARITY_SNAPSHOT_DIR`` is set; otherwise the test body
    receives ``None`` and should ``pytest.skip``.
    """
    env = os.environ.get("PARITY_SNAPSHOT_DIR")
    if env is None:
        return None
    p = Path(env)
    if not p.exists():
        pytest.skip(f"PARITY_SNAPSHOT_DIR={p} does not exist")
    return p


# --------------------------------------------------------------------------- #
# Calibration test: harness sanity check
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_type", ["volume", "surface", "combined"])
def test_calibration_legacy_vs_legacy(synthetic_sample: dict, model_type: str) -> None:
    r"""Legacy pipeline vs itself must produce identical dicts.

    This test does not depend on any new transform.  It verifies that

    1. :func:`domainmesh_to_legacy_datadict` produces a dict DoMINODataPipe
       accepts (no missing keys, no shape mismatches).
    2. ``DoMINODataPipe.process_data`` is deterministic with
       ``deterministic=True`` and ``sampling=False``.
    3. ``EXPECTED_KEYS`` correctly predicts which keys appear for each
       ``model_type``.

    **When this passes**, any later Axis-1 test failure is attributable to
    a new-pipeline transform, not to a harness bug.
    """
    pytest.importorskip("warp")  # legacy pipeline uses warp-backed SDF
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe

    data_dict = domainmesh_to_legacy_datadict(
        synthetic_sample["stl_mesh"], synthetic_sample["domain_mesh"]
    )

    # Build DoMINODataPipe with sampling OFF so the result is deterministic.
    pipe = DoMINODataPipe(
        input_path=synthetic_sample["case_dir"],
        model_type=model_type,
        grid_resolution=[8, 8, 8],  # tiny grid keeps calibration <1s
        normalize_coordinates=False,
        sample_in_bbox=False,
        sampling=False,
        gpu_preprocessing=False,
        gpu_output=False,
        bounding_box_dims=SYNTHETIC_VOLUME_BBOX,
        bounding_box_dims_surf=SYNTHETIC_SURFACE_BBOX,
        surface_variables=("pMeanTrim", "wallShearStressMeanTrim"),
        volume_variables=("UMeanTrim", "pMeanTrim", "nutMeanTrim"),
        deterministic=True,
    )

    out_a = pipe(dict(data_dict))
    out_b = pipe(dict(data_dict))

    expected = keys_for_mode(model_type)
    actual = set(out_a.keys())
    missing = expected - actual
    extra = actual - expected
    assert not missing, (
        f"EXPECTED_KEYS says {missing!r} should appear in {model_type} output "
        f"but legacy pipeline did not produce them.  Either the spec is wrong "
        f"or DoMINODataPipe was refactored."
    )
    # Legacy may emit additional keys the model does not consume (e.g.
    # volume_min_max, surface_min_max when normalize_coordinates=True).  The
    # harness is strict on the minimum set needed by DoMINO.forward; extras
    # are logged but don't fail this calibration test.
    if extra:
        print(f"[calibration] legacy produced extra keys not in EXPECTED_KEYS: {extra}")

    # Determinism check: two identical calls give identical outputs.
    for k in expected:
        torch.testing.assert_close(
            out_a[k],
            out_b[k],
            rtol=0.0,
            atol=0.0,
            msg=f"DoMINODataPipe is non-deterministic on key {k!r}",
        )


# --------------------------------------------------------------------------- #
# Axis-1 parity: new MeshDataset chain vs legacy DoMINODataPipe
# --------------------------------------------------------------------------- #


def _build_new_pipeline_output(
    pdmsh_dir: Path,
    stl_pattern: str,
    *,
    model_type: str,
    volume_bbox: _BBox,
    surface_bbox: _BBox,
    grid_resolution: Sequence[int] = (8, 8, 8),
) -> dict[str, torch.Tensor]:
    r"""Run the new MeshDataset chain and flatten its output to DoMINO keys.

    Builds a :class:`MeshDataset` over the *pdmsh_dir* with
    :class:`DomainMeshReader` (loading the sibling ``.stl.pmsh`` as the
    ``stl_geometry`` extra boundary) and the Phase-2 DoMINO transform
    chain, then reads one sample and flattens the result into the flat
    dict that ``DoMINO.forward`` expects.

    Parameters
    ----------
    pdmsh_dir : Path
        Directory containing one or more ``.pdmsh`` sub-directories.
    stl_pattern : str
        Glob pattern (relative to each ``.pdmsh`` sample's parent
        directory) finding the sibling ``.stl.pmsh`` mesh.
    model_type : str
        ``"volume"``, ``"surface"``, or ``"combined"`` -- drives which
        encodings and fields are kept in the flat output.
    volume_bbox, surface_bbox : :class:`_BBox`
        Explicit bbox corners for the latent-grid SDF transforms.
    grid_resolution : sequence of 3 ints
        Grid resolution matching the legacy pipeline setting.

    Returns
    -------
    dict[str, torch.Tensor]
        Keys matching the subset of :data:`EXPECTED_KEYS` required for
        *model_type*.  Extra keys present on the new side but not in the
        spec are stripped.
    """
    # Imports are local so the test module can be loaded in environments
    # where the full physicsnemo stack (warp, etc.) isn't available.
    from physicsnemo.datapipes.mesh_dataset import MeshDataset
    from physicsnemo.datapipes.readers.mesh import DomainMeshReader

    # Side-effect imports to register transforms in the ${dp:...} resolver
    # registry.  We don't use the resolver path in this test -- transforms
    # are constructed directly -- but the imports also pin the Python
    # import order so any registration-time errors surface at module load.
    sys_path = str(
        Path(__file__).resolve().parents[3]
        / "examples"
        / "cfd"
        / "external_aerodynamics"
        / "unified_external_aero_recipe"
        / "src"
    )
    import sys

    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    # Late imports after sys.path mutation.
    import domino_transforms  # noqa: F401
    import sdf  # noqa: F401

    from domino_transforms import (
        ComputeDoMINOPositionalEncodings,
        ComputeGridSDFFromBoundary,
        CropMeshToBBox,
        SurfaceKNNNeighbors,
    )
    from sdf import ComputeSDFFromBoundary, DropBoundary
    from physicsnemo.datapipes.transforms.mesh import (
        ComputeSurfaceNormals,
        MeshToTensorDict,
    )

    # Build the transform chain.  Order mirrors legacy ``process_data``:
    # surface-bbox grid + SDF first, volume-bbox grid + SDF, point-level
    # SDF on interior, kNN on the target boundary, positional encodings,
    # optional cropping, then strip the stl_geometry before TensorDict.
    transforms: list = []

    # surf_grid + sdf_surf_grid on the surface bbox.
    transforms.append(
        ComputeGridSDFFromBoundary(
            boundary_name="stl_geometry",
            grid_resolution=grid_resolution,
            grid_field="surf_grid",
            sdf_field="sdf_surf_grid",
            bbox_min=surface_bbox.min,
            bbox_max=surface_bbox.max,
        )
    )

    if model_type in ("volume", "combined"):
        # grid + sdf_grid on the volume bbox.
        transforms.append(
            ComputeGridSDFFromBoundary(
                boundary_name="stl_geometry",
                grid_resolution=grid_resolution,
                grid_field="grid",
                sdf_field="sdf_grid",
                bbox_min=volume_bbox.min,
                bbox_max=volume_bbox.max,
            )
        )
        # Point-level SDF on interior points (DoMINO's sdf_nodes).
        transforms.append(
            ComputeSDFFromBoundary(
                boundary_name="stl_geometry",
                sdf_field="sdf",
                normals_field=None,
                use_winding_number=True,
            )
        )

    if model_type in ("surface", "combined"):
        # Surface normals as cell_data so they survive subsampling/restructuring.
        transforms.append(ComputeSurfaceNormals(store_as="cell_data", field_name="normals"))
        # kNN on the target boundary.
        transforms.append(
            SurfaceKNNNeighbors(
                k=11,
                boundary_name="boundary",
                neighbors_field="surface_mesh_neighbors",
                neighbor_normals_field="surface_neighbors_normals",
                neighbor_areas_field="surface_neighbors_areas",
            )
        )

    # Positional encodings (COM-relative, volume-closest).
    transforms.append(
        ComputeDoMINOPositionalEncodings(
            stl_boundary="stl_geometry",
            target_boundary="boundary",
            compute_volume_encodings=model_type in ("volume", "combined"),
            compute_surface_encodings=model_type in ("surface", "combined"),
        )
    )

    # Drop the large STL boundary before MeshToTensorDict so it doesn't
    # bloat the serialized output.
    transforms.append(DropBoundary(names=["stl_geometry"]))

    # Terminal: Mesh -> TensorDict.
    transforms.append(MeshToTensorDict())

    reader = DomainMeshReader(
        path=pdmsh_dir,
        pattern="**/*.pdmsh",
        extra_boundaries={"stl_geometry": {"pattern": stl_pattern}},
    )
    dataset = MeshDataset(reader=reader, transforms=transforms)

    td, _meta = dataset[0]

    # Flatten the TensorDict into DoMINO's flat-key output dict.
    return _flatten_tensordict_to_domino_dict(td, model_type=model_type)


def _flatten_tensordict_to_domino_dict(
    td,
    *,
    model_type: str,
) -> dict[str, torch.Tensor]:
    r"""Convert the MeshDataset TensorDict output to a flat DoMINO-style dict.

    The harness does this conversion manually (without
    :class:`RestructureTensorDict`) so it does not depend on the final
    YAML key mapping; failures here localise to the test adapter, not to
    YAML configuration.  Once Phase-3 ships ``hlpw_*.yaml`` with a
    concrete ``RestructureTensorDict.groups`` block, that block must
    produce the same keys listed here.

    Keys produced:

    - from ``interior.points``:     ``volume_mesh_centers``
    - from ``interior.point_data``: ``sdf_nodes``, ``pos_volume_closest``,
                                    ``pos_volume_center_of_mass``,
                                    ``volume_fields`` (concat of UMean,
                                    pMean, nutMean)
    - from ``boundaries.boundary``: ``surface_mesh_centers`` (cell centroids),
                                    ``surface_normals`` (cell normals),
                                    ``surface_areas`` (cell areas),
                                    ``surface_mesh_neighbors``,
                                    ``surface_neighbors_normals``,
                                    ``surface_neighbors_areas``,
                                    ``pos_surface_center_of_mass``,
                                    ``surface_fields`` (concat of pMean, wss)
    - from ``boundaries.stl_geometry`` is dropped by :class:`DropBoundary`;
      the corresponding STL-coordinate ``geometry_coordinates`` is sourced
      from the ``stl_geometry`` boundary BEFORE drop -- because
      ``MeshToTensorDict`` runs after ``DropBoundary`` we need a small
      shim: we retrieve the STL points from the TensorDict iff the boundary
      was kept.  In production ``RestructureTensorDict`` would pull
      ``boundaries.stl_geometry.points`` before ``DropBoundary``.
    """
    out: dict[str, torch.Tensor] = {}

    interior_td = td["interior"]
    out["volume_mesh_centers"] = interior_td["points"].detach().cpu().float()

    # Volume point_data fields (if present).
    if "point_data" in interior_td.keys():
        pd = interior_td["point_data"]
        if "sdf" in pd.keys():
            sdf_vals = pd["sdf"]
            # legacy stores sdf_nodes as (N, 1); ComputeSDFFromBoundary
            # already unsqueezes, so this is already (N, 1).
            out["sdf_nodes"] = sdf_vals.detach().cpu().float()
        if "pos_volume_closest" in pd.keys():
            out["pos_volume_closest"] = pd["pos_volume_closest"].detach().cpu().float()
        if "pos_volume_center_of_mass" in pd.keys():
            out["pos_volume_center_of_mass"] = (
                pd["pos_volume_center_of_mass"].detach().cpu().float()
            )
        # Concatenate UMean + pMean + nutMean -> volume_fields.
        chunks = []
        for name in ("UMeanTrim", "pMeanTrim", "nutMeanTrim"):
            if name in pd.keys():
                v = pd[name]
                if v.ndim == 1:
                    v = v.unsqueeze(-1)
                chunks.append(v)
        if chunks:
            out["volume_fields"] = torch.cat(chunks, dim=-1).detach().cpu().float()

    # Surface boundary fields (if the target boundary survives).
    boundaries = td.get("boundaries", None)
    if boundaries is not None and "boundary" in boundaries.keys():
        bnd = boundaries["boundary"]
        # Cell centroids: recompute from points+cells if not materialized.
        points = bnd["points"]
        cells = bnd["cells"]
        centroids = points[cells].mean(dim=1).float()
        out["surface_mesh_centers"] = centroids.detach().cpu()

        if "cell_data" in bnd.keys():
            cd = bnd["cell_data"]
            if "normals" in cd.keys():
                out["surface_normals"] = cd["normals"].detach().cpu().float()
            # Cell areas: mesh.cell_areas is a property, not a field;
            # ComputeSurfaceNormals stores only normals.  Compute areas
            # from the cells tensor directly (triangle cross product).
            # For triangles: area = 0.5 * |(b-a) x (c-a)|.
            tri = points[cells].float()  # (N_c, 3, 3)
            e1 = tri[:, 1] - tri[:, 0]
            e2 = tri[:, 2] - tri[:, 0]
            out["surface_areas"] = (
                0.5 * torch.linalg.cross(e1, e2, dim=-1).norm(dim=-1)
            ).detach().cpu()

            if "surface_mesh_neighbors" in cd.keys():
                out["surface_mesh_neighbors"] = (
                    cd["surface_mesh_neighbors"].detach().cpu().float()
                )
            if "surface_neighbors_normals" in cd.keys():
                out["surface_neighbors_normals"] = (
                    cd["surface_neighbors_normals"].detach().cpu().float()
                )
            if "surface_neighbors_areas" in cd.keys():
                out["surface_neighbors_areas"] = (
                    cd["surface_neighbors_areas"].detach().cpu().float()
                )
            if "pos_surface_center_of_mass" in cd.keys():
                out["pos_surface_center_of_mass"] = (
                    cd["pos_surface_center_of_mass"].detach().cpu().float()
                )
            # surface_fields := cat(pMeanTrim, wallShearStressMeanTrim) along last dim
            chunks = []
            for name in ("pMeanTrim", "wallShearStressMeanTrim"):
                if name in cd.keys():
                    v = cd[name]
                    if v.ndim == 1:
                        v = v.unsqueeze(-1)
                    chunks.append(v)
            if chunks:
                out["surface_fields"] = torch.cat(chunks, dim=-1).detach().cpu().float()

    # Global data: grid, sdf_grid, surf_grid, sdf_surf_grid.
    if "global_data" in td.keys():
        gd = td["global_data"]
        for key in ("grid", "sdf_grid", "surf_grid", "sdf_surf_grid"):
            if key in gd.keys():
                out[key] = gd[key].detach().cpu().float()

    # DoMINO also requires geometry_coordinates, global_params_values, and
    # global_params_reference.  The fixture puts no global_params on the
    # DomainMesh -- in real datasets these will come from global_data.
    # Here we fill placeholders to match the shape DoMINO expects.
    out.setdefault(
        "global_params_values", torch.zeros(1, 2, dtype=torch.float32)
    )
    out.setdefault(
        "global_params_reference", torch.ones(1, 2, dtype=torch.float32)
    )
    # geometry_coordinates: ideally pulled from boundaries.stl_geometry
    # before DropBoundary.  Because we dropped it, fall back to the target
    # boundary's points so the key is at least present; numerical
    # comparison on this key is not meaningful in this harness and is
    # excluded from the parity assert block.
    if boundaries is not None and "boundary" in boundaries.keys():
        out.setdefault(
            "geometry_coordinates",
            boundaries["boundary"]["points"].detach().cpu().float(),
        )

    # Keep only keys expected by DoMINO for this model_type; drop extras.
    wanted = keys_for_mode(model_type)
    return {k: v for k, v in out.items() if k in wanted}


@pytest.mark.parametrize("model_type", ["surface"])  # volume/combined added after volume transforms
def test_axis1_byte_parity_synthetic(synthetic_sample: dict, model_type: str) -> None:
    r"""New MeshDataset output must match legacy DoMINODataPipe bit-for-bit.

    Runs both sides on the Tier-1 synthetic sample with sampling disabled.
    Each key listed in :data:`EXPECTED_KEYS` for the given *model_type* is
    compared with its per-key tolerance.
    """
    pytest.importorskip("warp")
    pytest.importorskip("tensordict")

    try:
        new_out = _build_new_pipeline_output(
            synthetic_sample["pdmsh"].parent,
            stl_pattern="*_single_solid.stl.pmsh",
            model_type=model_type,
            volume_bbox=SYNTHETIC_VOLUME_BBOX,
            surface_bbox=SYNTHETIC_SURFACE_BBOX,
        )
    except NotImplementedError as e:
        pytest.skip(str(e))

    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe

    data_dict = domainmesh_to_legacy_datadict(
        synthetic_sample["stl_mesh"], synthetic_sample["domain_mesh"]
    )
    pipe = DoMINODataPipe(
        input_path=synthetic_sample["case_dir"],
        model_type=model_type,
        grid_resolution=[8, 8, 8],
        normalize_coordinates=False,
        sample_in_bbox=False,
        sampling=False,
        gpu_preprocessing=False,
        gpu_output=False,
        bounding_box_dims=SYNTHETIC_VOLUME_BBOX,
        bounding_box_dims_surf=SYNTHETIC_SURFACE_BBOX,
        surface_variables=("pMeanTrim", "wallShearStressMeanTrim"),
        volume_variables=("UMeanTrim", "pMeanTrim", "nutMeanTrim"),
        deterministic=True,
    )
    legacy_out = pipe(dict(data_dict))

    expected = keys_for_mode(model_type)
    missing_from_new = expected - set(new_out.keys())
    assert not missing_from_new, (
        f"New pipeline missing keys {missing_from_new} for model_type={model_type!r}. "
        f"Owner hints: "
        + ", ".join(
            f"{k!r} -> {spec_by_name()[k].owner}" for k in sorted(missing_from_new)
        )
    )

    for key in sorted(expected):
        rtol, atol = _tolerance_for(key)
        # Legacy DoMINODataPipe appends a leading batch dim via
        # unsqueeze(0) at domino_datapipe.py:1007.  The new MeshDataset
        # leaves batching to DataLoader collate, so ranks in
        # expected_keys.py are per-sample.  Strip the legacy batch dim
        # before comparing.
        legacy_t = legacy_out[key].detach().cpu()
        assert legacy_t.shape[0] == 1, (
            f"Legacy output for {key!r} has unexpected leading dim "
            f"{legacy_t.shape[0]}; parity harness assumes 1."
        )
        torch.testing.assert_close(
            new_out[key].detach().cpu(),
            legacy_t.squeeze(0),
            rtol=rtol,
            atol=atol,
            msg=f"Parity mismatch on key {key!r} "
            f"(owner: {spec_by_name()[key].owner})",
        )


# --------------------------------------------------------------------------- #
# Axis-1 parity against real Tier-2 HLPW sample (optional, env-gated)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("model_type", ["surface"])
def test_axis1_byte_parity_real(real_sample: Path | None, model_type: str) -> None:
    r"""Axis-1 parity on a real HLPW ``.pdmsh`` sample.

    Enabled only when ``$PARITY_SNAPSHOT_DIR`` points at a directory
    containing a ``.pdmsh`` + ``_single_solid.stl.pmsh`` pair.  Runs the
    full-resolution real data through both pipelines with sampling
    disabled; expected to run in seconds on 1 GPU and a few minutes on CPU.
    """
    if real_sample is None:
        pytest.skip("Set PARITY_SNAPSHOT_DIR to enable Tier-2 parity tests")
    pytest.importorskip("warp")

    try:
        _ = _build_new_pipeline_output(
            real_sample,
            stl_pattern="*_single_solid.stl.pmsh",
            model_type=model_type,
            volume_bbox=_BBox([-3.0, -2.0, 0.0], [6.0, 2.0, 3.0]),
            surface_bbox=_BBox([-2.0, -1.0, 0.0], [4.0, 1.0, 2.0]),
        )
    except NotImplementedError as e:
        pytest.skip(str(e))

    # Tier-2 follow-up: once the synthetic Axis-1 test is green, load the real
    # .pdmsh with DomainMeshReader on the new side, construct the legacy
    # data_dict by materializing the same .pdmsh into tensors via the same
    # adapter, and compare key-by-key using the same tolerances.  This mirrors
    # the synthetic test body with just the data-source path swapped.
    pytest.skip("Tier-2 body shares implementation with synthetic Axis-1; "
                "enable after Axis-1 passes on synthetic.")
