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
Synthetic DomainMesh fixture for DoMINO parity testing.

Builds a small deterministic automotive-like :class:`DomainMesh` (icosphere
surface boundary + random interior point cloud with per-point CFD fields) and
writes it to disk as a ``.pdmsh`` directory plus a sibling ``.stl.pmsh``
directory.  Both the legacy ``PmshFileReader`` path and the new
``DomainMeshReader`` path read from the same directory.

Design goals
------------
- Fits in <5 MB on disk (target: ~500-triangle sphere, 10k interior points)
- Regenerable bit-for-bit from ``seed`` (no non-determinism)
- Non-trivial bounding box (x in [-2, 2], y in [-1, 1], z in [0, 1]) so that
  centering + bbox normalization transforms exercise non-zero offsets.
- Surface boundary has ``cell_data`` fields matching legacy DoMINO surface
  variables (``pMean``, ``wallShearStress``).
- Interior has ``point_data`` fields matching legacy DoMINO volume variables
  (``UMean``, ``pMean``, ``nutMean``).
- ``global_data`` carries ``U_inf``, ``p_inf``, ``rho_inf``, ``nu``, ``L_ref``
  so ``NonDimensionalizeByMetadata`` works unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from physicsnemo.mesh import DomainMesh, Mesh  # noqa: F401  type hints only


def _icosphere(
    subdivisions: int = 2,
    radius: float = 0.5,
    center: tuple[float, float, float] = (0.0, 0.0, 0.5),
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Return ``(points, cells)`` of an icosphere mesh.

    Subdivides each triangle of the regular icosahedron *subdivisions* times
    (each subdivision multiplies face count by 4).  Vertices are normalized
    to the sphere of the given radius around *center*.

    ``subdivisions=2`` gives 162 vertices and 320 triangles.
    ``subdivisions=3`` gives 642 vertices and 1280 triangles.

    Parameters
    ----------
    subdivisions : int
        Number of midpoint-refinement subdivisions (>=0).
    radius : float
        Target radius.
    center : tuple of float
        Sphere centre in world coordinates.
    dtype : torch.dtype
        Tensor element dtype for points.

    Returns
    -------
    points : Tensor, shape (n_points, 3)
        Vertex coordinates.
    cells : Tensor, shape (n_triangles, 3), dtype int64
        Triangle connectivity.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base_verts = torch.tensor(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=dtype,
    )
    base_cells = torch.tensor(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=torch.int64,
    )

    points = base_verts
    cells = base_cells

    for _ in range(subdivisions):
        midpoint_cache: dict[tuple[int, int], int] = {}
        new_cells = []
        new_points = list(points)

        def midpoint(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key in midpoint_cache:
                return midpoint_cache[key]
            mp = (new_points[a] + new_points[b]) / 2.0
            new_points.append(mp)
            idx = len(new_points) - 1
            midpoint_cache[key] = idx
            return idx

        for tri in cells.tolist():
            a, b, c = tri
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_cells.extend(
                [
                    [a, ab, ca],
                    [b, bc, ab],
                    [c, ca, bc],
                    [ab, bc, ca],
                ]
            )

        points = torch.stack(new_points, dim=0)
        cells = torch.tensor(new_cells, dtype=torch.int64)

    # Project to sphere of given radius, translate to centre.
    norms = points.norm(dim=-1, keepdim=True)
    points = points / norms * radius
    points = points + torch.tensor(center, dtype=dtype)
    return points, cells


def _shuffle_mesh(
    points: torch.Tensor,
    cells: torch.Tensor,
    cell_data: dict[str, torch.Tensor] | None,
    point_data: dict[str, torch.Tensor] | None,
    generator: torch.Generator,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor] | None,
    dict[str, torch.Tensor] | None,
]:
    r"""Shuffle *points* and *cells* in parallel with their data tensors.

    Mirrors the contract required by the PR's contiguous-block sampler:
    on-disk point order must be randomized so a contiguous slice is spatially
    representative, and cell order must be randomized so a contiguous cell
    slice is also spatially representative.

    Parameters
    ----------
    points : Tensor, shape (N_p, 3)
    cells : Tensor, shape (N_c, K) or None
        Triangle connectivity indexing into ``points``.
    cell_data : dict[str, Tensor] or None
        Per-cell fields to permute along with ``cells``.
    point_data : dict[str, Tensor] or None
        Per-point fields to permute along with ``points``.
    generator : torch.Generator
        RNG (advances state in place so successive calls are independent).

    Returns
    -------
    tuple of permuted ``(points, cells, cell_data, point_data)``.
    """
    n_points = points.shape[0]
    perm_p = torch.randperm(n_points, generator=generator)
    inv_p = torch.empty_like(perm_p)
    inv_p[perm_p] = torch.arange(n_points)
    new_points = points[perm_p]

    new_point_data: dict[str, torch.Tensor] | None = None
    if point_data is not None:
        new_point_data = {k: v[perm_p] for k, v in point_data.items()}

    new_cells: torch.Tensor | None = None
    new_cell_data: dict[str, torch.Tensor] | None = None
    if cells is not None:
        new_cells = inv_p[cells]
        n_cells = new_cells.shape[0]
        perm_c = torch.randperm(n_cells, generator=generator)
        new_cells = new_cells[perm_c]
        if cell_data is not None:
            new_cell_data = {k: v[perm_c] for k, v in cell_data.items()}

    return new_points, new_cells, new_cell_data, new_point_data


def build_synthetic_sample(
    seed: int = 42,
    n_interior: int = 10_000,
    surface_subdivisions: int = 2,
    shuffle_on_disk: bool = True,
) -> tuple["Mesh", "DomainMesh"]:
    r"""Build a synthetic ``(stl_mesh, domain_mesh)`` pair.

    Deterministic given *seed*.

    Parameters
    ----------
    seed : int
        RNG seed for all stochastic steps (interior point placement,
        random fields, on-disk shuffling).
    n_interior : int
        Number of interior point-cloud points in the domain mesh.
    surface_subdivisions : int
        Icosphere subdivision level for the surface boundary (also reused as
        the STL geometry).  ``2`` is ~320 triangles (<5 MB fixture target).
    shuffle_on_disk : bool
        If True, random-permute points and cells before writing so that the
        PR's contiguous-block sampler behaves like uniform random sampling.
        Set False only when you want to probe sampler behaviour on
        adversarially-ordered data.

    Returns
    -------
    stl_mesh : Mesh
        Surface mesh in STL role (used as ``extra_boundaries.stl_geometry``).
    domain_mesh : DomainMesh
        Full domain mesh with ``interior``, ``boundaries["boundary"]``, and
        ``global_data``.
    """
    # Import here so the module is importable in environments without warp
    # (e.g. login-node lint/type-check).  Tests that actually call this
    # function run inside the container, where the imports succeed.
    from physicsnemo.mesh import DomainMesh, Mesh

    g = torch.Generator().manual_seed(seed)

    # 1) Surface geometry (shared between .stl.pmsh and boundary).
    surf_points, surf_cells = _icosphere(
        subdivisions=surface_subdivisions, radius=0.5, center=(0.0, 0.0, 0.5)
    )

    # Per-cell "CFD-like" fields: pressure (scalar) + wall shear stress (3-vec).
    n_tri = surf_cells.shape[0]
    cell_centroids = surf_points[surf_cells].mean(dim=1)
    # Pressure: a smooth function of centroid z so the test can detect
    # mis-ordered indexing.
    p_mean = (cell_centroids[:, 2] - 0.5).unsqueeze(-1) * 0.1  # (n_tri, 1)
    wss_mean = 0.01 * torch.randn(n_tri, 3, generator=g)

    surf_cell_data: dict[str, torch.Tensor] = {
        "pMeanTrim": p_mean,
        "wallShearStressMeanTrim": wss_mean,
    }

    # 2) Interior point cloud in a box around and above the surface.
    bbox_min = torch.tensor([-2.0, -1.0, 0.0])
    bbox_max = torch.tensor([2.0, 1.0, 2.0])
    u = torch.rand(n_interior, 3, generator=g)
    interior_points = bbox_min + u * (bbox_max - bbox_min)

    # Per-point CFD-like fields.
    u_mean = 30.0 * torch.stack(
        [
            1.0 + 0.1 * torch.sin(interior_points[:, 0]),
            0.1 * torch.randn(n_interior, generator=g),
            0.1 * torch.randn(n_interior, generator=g),
        ],
        dim=-1,
    )  # (n_interior, 3)
    p_vol = 0.5 * (interior_points[:, 0] ** 2 + interior_points[:, 1] ** 2).unsqueeze(-1)
    nut_vol = 1e-4 * torch.rand(n_interior, 1, generator=g)

    interior_point_data: dict[str, torch.Tensor] = {
        "UMeanTrim": u_mean,
        "pMeanTrim": p_vol,
        "nutMeanTrim": nut_vol,
    }

    # 3) Global metadata (freestream conditions + reference length).
    # Shapes chosen to exercise the scalar and vector branches of
    # NonDimensionalizeByMetadata.
    global_data: dict[str, torch.Tensor] = {
        "U_inf": torch.tensor([30.0, 0.0, 0.0]),
        "p_inf": torch.tensor(0.0),
        "rho_inf": torch.tensor(1.225),
        "nu": torch.tensor(1.5e-5),
        "L_ref": torch.tensor(5.0),
    }

    # 4) Optionally shuffle on-disk ordering so the PR's contiguous-block
    # sampler behaves like uniform sampling.
    if shuffle_on_disk:
        (
            surf_points,
            surf_cells,
            surf_cell_data,
            _,  # no point_data on surface
        ) = _shuffle_mesh(surf_points, surf_cells, surf_cell_data, None, g)
        (
            interior_points,
            _,  # no cells on interior
            _,
            interior_point_data,
        ) = _shuffle_mesh(interior_points, None, None, interior_point_data, g)

    # 5) Assemble Mesh / DomainMesh objects.
    # STL role: geometry only, no fields.
    stl_mesh = Mesh(
        points=surf_points.clone(),
        cells=surf_cells.clone(),
    )

    boundary = Mesh(
        points=surf_points,
        cells=surf_cells,
        cell_data=surf_cell_data,
    )

    interior = Mesh(
        points=interior_points,
        point_data=interior_point_data,
    )

    domain = DomainMesh(
        interior=interior,
        boundaries={"boundary": boundary},
        global_data=global_data,
    )

    return stl_mesh, domain


def write_synthetic_sample(
    out_dir: Path | str,
    seed: int = 42,
    n_interior: int = 10_000,
    surface_subdivisions: int = 2,
    shuffle_on_disk: bool = True,
    case_name: str = "case_000",
) -> dict[str, Path]:
    r"""Build and persist one synthetic sample to disk.

    Layout produced::

        out_dir/
          case_000/
            domain_000.pdmsh/     (DomainMesh with interior + 'boundary')
            case_000_single_solid.stl.pmsh/   (Mesh for stl_geometry role)

    Both directories are tensordict memmap archives.  Loading them back with
    ``DomainMesh.load`` / ``Mesh.load`` recovers the in-memory tensors
    bit-for-bit (subject to float32 precision).

    Parameters
    ----------
    out_dir : Path or str
        Parent directory to write into.  Created if missing.
    seed, n_interior, surface_subdivisions, shuffle_on_disk
        Forwarded to :func:`build_synthetic_sample`.
    case_name : str
        Sub-directory name.  Defaults to ``case_000``.

    Returns
    -------
    dict[str, Path]
        ``{"case_dir", "pdmsh", "stl_pmsh"}`` absolute paths.
    """
    out_dir = Path(out_dir)
    case_dir = out_dir / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    stl_mesh, domain_mesh = build_synthetic_sample(
        seed=seed,
        n_interior=n_interior,
        surface_subdivisions=surface_subdivisions,
        shuffle_on_disk=shuffle_on_disk,
    )

    pdmsh_path = case_dir / "domain_000.pdmsh"
    stl_path = case_dir / f"{case_name}_single_solid.stl.pmsh"

    domain_mesh.save(str(pdmsh_path))
    stl_mesh.save(str(stl_path))

    return {
        "case_dir": case_dir.resolve(),
        "pdmsh": pdmsh_path.resolve(),
        "stl_pmsh": stl_path.resolve(),
    }
