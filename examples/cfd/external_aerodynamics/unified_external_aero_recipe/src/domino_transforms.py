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
DoMINO-specific mesh transforms for the unified external-aerodynamics recipe.

These transforms fill the gap between PR #1512's library-level
:mod:`physicsnemo.datapipes.transforms.mesh` (generic mesh operations) and
what :class:`~physicsnemo.models.domino.model.DoMINO.forward` consumes.  They
are **recipe-local** -- same placement convention used by ``sdf.py`` and
``nondim.py`` in this directory.  Importing this module registers all
classes into the global datapipe component registry so they are reachable
as ``${dp:Name}`` in Hydra configs.

Transforms provided
-------------------

- :class:`ComputeGridSDFFromBoundary` -- 3D latent-grid SDF (DoMINO's
  ``grid`` + ``sdf_grid`` / ``surf_grid`` + ``sdf_surf_grid``).  Distinct
  from the library's :class:`~sdf.ComputeSDFFromBoundary`, which operates
  point-wise on interior points.  Supports ``normalize_coords`` to compute
  the SDF in the :math:`[-1, 1]` normalized frame (legacy DoMINO behaviour).
- :class:`SurfaceKNNNeighbors` -- wraps :func:`physicsnemo.nn.functional.knn`
  (cuml-accelerated on CUDA via cupy dlpack, scipy on CPU) to compute
  ``surface_mesh_neighbors``, ``surface_neighbors_normals``,
  ``surface_neighbors_areas`` as ``cell_data`` fields.
- :class:`ComputeDoMINOPositionalEncodings` -- computes the three DoMINO
  positional encoding tensors (``pos_surface_center_of_mass``,
  ``pos_volume_closest``, ``pos_volume_center_of_mass``).
- :class:`CropMeshToBBox` -- drop interior points outside a bounding box
  (with matching ``point_data`` filtering).
- :class:`SubsampleNamedBoundary` -- random-without-replacement subsample
  of a named :class:`DomainMesh` boundary's points via
  :func:`physicsnemo.datapipes.transforms.subsample.shuffle_array`.
  Used for STL geometry downsampling (legacy ``geom_points_sample``).
- :class:`LiftCellCentroidsAndAreas` -- materialize cell centroids and
  cell areas of a named boundary as explicit ``cell_data`` tensors.
- :class:`PromoteBoundaryToInterior` -- expose a boundary's cell-centered
  fields through ``interior.point_data`` so the unified trainer can extract
  targets without a flat TensorDict projection.
- :class:`NormalizeDomainByBBox` -- rescale every spatial tensor in a
  :class:`DomainMesh` (interior points, named boundary points, cell
  centroids, selected ``global_data`` tensors such as ``grid`` and
  ``surf_grid``) to :math:`[-1, 1]` using explicit bbox corners.  Emits
  ``surface_min_max`` and ``volume_min_max`` into ``global_data`` so the
  model can denormalize predictions downstream.  Supersedes the older
  ``NormalizeCoordinatesByBBox`` (removed in this revision).

Design notes
------------

- All transforms subclass :class:`MeshTransform` and override
  :meth:`apply_to_domain` so :class:`MeshDataset` routes correctly for
  :class:`DomainMesh` inputs.  The plain :meth:`__call__` path is defined
  for single-:class:`Mesh` inputs where the transform is meaningful, and
  raises otherwise.
- Every transform returns a **new** :class:`Mesh` / :class:`DomainMesh`;
  none mutates in place.  This is required for the dataset's prefetch
  cache to remain safe under multi-threaded access.
- Fields computed here go into the mesh's ``global_data``, ``point_data``,
  or ``cell_data`` depending on their shape.  The recipe's declarative
  ``forward_kwargs`` spec then maps those DomainMesh paths to the dict that
  DoMINO consumes.
- Tensors are kept on the same device as the input mesh; callers
  (``MeshDataset``) are responsible for device placement.

References
----------
Legacy pipeline equivalents live in
``physicsnemo/datapipes/cae/domino_datapipe.py`` in
``DoMINODataPipe.process_data``, ``process_surface``, ``process_volume``.
"""

from __future__ import annotations

from typing import Literal, Sequence

import torch
from jaxtyping import Float, Int

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.datapipes.transforms.subsample import shuffle_array
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.models.domino.utils.utils import (
    calculate_center_of_mass,
    create_grid,
)
from physicsnemo.nn.functional import knn, signed_distance_field


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tensor_from_bbox(
    bbox: Sequence[float] | torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    r"""Normalize a ``bbox_min`` / ``bbox_max`` input to a device-placed tensor.

    Accepts either an iterable of three floats (e.g. from YAML) or a preexisting
    tensor, and returns a ``(3,)`` tensor on the requested device+dtype.
    """
    t = torch.as_tensor(bbox, dtype=dtype, device=device)
    if t.shape != (3,):
        raise ValueError(f"Expected bbox of shape (3,), got {tuple(t.shape)}")
    return t


def _validate_bbox(
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    r"""Coerce ``(bbox_min, bbox_max)`` to length-3 float tuples, raising on bad input.

    Used by transforms that accept both corners via kwargs.  Centralized here
    so constructor error messages stay consistent across the module.
    """
    mn = tuple(float(x) for x in bbox_min)
    mx = tuple(float(x) for x in bbox_max)
    if len(mn) != 3 or len(mx) != 3:
        raise ValueError(f"bbox_min/bbox_max must be length 3, got {mn!r}, {mx!r}")
    return mn, mx  # type: ignore[return-value]


def _derive_bbox_from_points(
    points: Float[torch.Tensor, "n 3"],
    *,
    pad_fraction: float = 0.0,
) -> tuple[Float[torch.Tensor, "3"], Float[torch.Tensor, "3"]]:
    r"""Compute axis-aligned ``(min, max)`` from a point cloud.

    Parameters
    ----------
    points : Tensor, shape (N, 3)
        Points to bound.
    pad_fraction : float
        Symmetric padding as a fraction of the span along each axis.
        ``0.0`` means tight bbox.  Positive values grow the bbox outward.

    Returns
    -------
    (bbox_min, bbox_max) : tuple of Tensor, each shape (3,)
    """
    bbox_min = points.amin(dim=0)
    bbox_max = points.amax(dim=0)
    if pad_fraction != 0.0:
        span = bbox_max - bbox_min
        bbox_min = bbox_min - pad_fraction * span
        bbox_max = bbox_max + pad_fraction * span
    return bbox_min, bbox_max


def _rescale_bbox(
    x: Float[torch.Tensor, "..."],
    bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
) -> Float[torch.Tensor, "..."]:
    r"""Apply :math:`2 (x - \min) / (\max - \min) - 1` broadcasting over trailing dim 3.

    Matches legacy ``physicsnemo.models.domino.utils.normalize`` (git sha
    ``d2eb6cc8``). The ``bbox`` argument is the pair of
    ``(min, max)`` length-3 float tuples returned by :func:`_validate_bbox`;
    tensors are instantiated on ``x``'s device and dtype per call so the
    helper is safe to share across transforms without carrying state.
    """
    mn = torch.tensor(bbox[0], dtype=x.dtype, device=x.device)
    mx = torch.tensor(bbox[1], dtype=x.dtype, device=x.device)
    span = (mx - mn).clamp(min=1e-12)
    return 2.0 * (x - mn) / span - 1.0


# --------------------------------------------------------------------------- #
# ComputeGridSDFFromBoundary
# --------------------------------------------------------------------------- #


@register()
class ComputeGridSDFFromBoundary(MeshTransform):
    r"""Compute signed distance field on a 3D latent grid around a surface.

    For DoMINO this produces the inputs ``grid`` / ``sdf_grid`` (for the
    volume geometry rep) and ``surf_grid`` / ``sdf_surf_grid`` (for the
    surface geometry rep).  The grid is built via
    :func:`physicsnemo.models.domino.utils.create_grid`, and
    :func:`physicsnemo.nn.functional.signed_distance_field` evaluates the
    SDF at every grid point using the Warp-backed BVH.

    Results are written into ``domain.global_data``:

    - ``{grid_field}``: Tensor of shape :math:`(n_x, n_y, n_z, 3)`
    - ``{sdf_field}``:  Tensor of shape :math:`(n_x, n_y, n_z)`

    Parameters
    ----------
    boundary_name : str
        Name of the :class:`DomainMesh` boundary to use as the closed
        surface.  Must have triangle cells.
    grid_resolution : sequence of 3 ints
        Number of samples along ``[nx, ny, nz]``.
    grid_field : str
        Global-data key for the 3D grid coordinates.
    sdf_field : str
        Global-data key for the SDF values on that grid.
    bbox_min, bbox_max : sequence of 3 floats, optional
        Explicit grid bounds.  If ``None`` (the default), both are derived
        from the boundary points' axis-aligned bounding box.
    use_winding_number : bool
        Whether to use winding-number sign computation.  Required for
        non-watertight meshes.  Default ``True``.
    normalize_coords : bool
        When ``True``, normalize both the STL boundary vertices and the
        generated grid into :math:`[-1, 1]` (using ``bbox_min`` / ``bbox_max``
        as the normalization range) *before* calling
        :func:`signed_distance_field`, and store the normalized grid back
        into ``global_data``.  Mirrors legacy ``DoMINODataPipe.process_data``
        (see ``physicsnemo/datapipes/cae/domino_datapipe.py`` lines 712-727
        at git sha ``d2eb6cc8``) which computes the grid SDF in the
        normalized frame.  Requires both ``bbox_min`` and ``bbox_max`` to
        be provided (raising :class:`ValueError` otherwise, since the
        data-derived bbox would collapse the normalization to a no-op).
        Default ``False``.

    Notes
    -----
    The output ``sdf_grid`` is shape ``(nx, ny, nz)`` (no channel dim) so
    that it can be consumed by
    :class:`~physicsnemo.models.domino.geometry_rep.geo_rep_volume` /
    ``geo_rep_surface`` without reshaping.  This matches the legacy
    ``DoMINODataPipe.process_data`` convention.
    """

    def __init__(
        self,
        *,
        boundary_name: str = "stl_geometry",
        grid_resolution: Sequence[int] = (256, 96, 64),
        grid_field: str = "grid",
        sdf_field: str = "sdf_grid",
        bbox_min: Sequence[float] | None = None,
        bbox_max: Sequence[float] | None = None,
        use_winding_number: bool = True,
        normalize_coords: bool = False,
    ) -> None:
        super().__init__()
        self.boundary_name = boundary_name
        # Store resolution as a plain tuple; cast to tensor at compute time
        # on the mesh's device.
        self.grid_resolution = tuple(int(x) for x in grid_resolution)
        if len(self.grid_resolution) != 3:
            raise ValueError(
                f"grid_resolution must be length 3, got {self.grid_resolution!r}"
            )
        self.grid_field = grid_field
        self.sdf_field = sdf_field
        self.bbox_min = (
            tuple(float(x) for x in bbox_min) if bbox_min is not None else None
        )
        self.bbox_max = (
            tuple(float(x) for x in bbox_max) if bbox_max is not None else None
        )
        self.use_winding_number = use_winding_number
        self.normalize_coords = bool(normalize_coords)
        if self.normalize_coords and (self.bbox_min is None or self.bbox_max is None):
            raise ValueError(
                "normalize_coords=True requires both bbox_min and bbox_max "
                "to be provided; data-derived bbox would make normalization "
                "a no-op."
            )

    def __call__(self, mesh: Mesh) -> Mesh:
        # Single-Mesh path: grid SDF needs a separate surface to query
        # against, so without a boundary reference this transform is a no-op.
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Compute grid + SDF on that grid, writing results to ``global_data``."""
        if self.boundary_name not in domain.boundaries:
            raise KeyError(
                f"Boundary {self.boundary_name!r} not found. "
                f"Available: {domain.boundary_names}"
            )
        surface = domain.boundaries[self.boundary_name]
        device = surface.points.device
        dtype = torch.float32  # Warp SDF kernels are float32

        if self.bbox_min is None or self.bbox_max is None:
            bbox_min_t, bbox_max_t = _derive_bbox_from_points(
                surface.points.to(dtype=dtype)
            )
        else:
            bbox_min_t = _tensor_from_bbox(self.bbox_min, device=device, dtype=dtype)
            bbox_max_t = _tensor_from_bbox(self.bbox_max, device=device, dtype=dtype)

        resolution = torch.tensor(
            self.grid_resolution, dtype=torch.int64, device=device
        )

        # create_grid returns (nx, ny, nz, 3) on the device of its inputs.
        grid = create_grid(bbox_max_t.to(device), bbox_min_t.to(device), resolution)

        # Legacy DoMINO computes grid SDF in the normalized [-1, 1] frame:
        # it rescales both the STL vertices and the grid into [-1, 1] via the
        # surface/volume bbox, runs signed_distance_field there, and stores the
        # normalized grid (see domino_datapipe.py l712-727 @ d2eb6cc8).  The SDF
        # magnitudes thus live in bbox-span units, not world units; downstream
        # scaling factors (``surface_sdf_scaling_factor``) assume this convention.
        sdf_points = surface.points.to(dtype=dtype)
        sdf_grid_coords = grid
        if self.normalize_coords:
            span = (bbox_max_t - bbox_min_t).clamp(min=1e-12)
            sdf_points = 2.0 * (sdf_points - bbox_min_t) / span - 1.0
            sdf_grid_coords = 2.0 * (grid - bbox_min_t) / span - 1.0
            stored_grid = sdf_grid_coords
        else:
            stored_grid = grid

        sdf_grid, _ = signed_distance_field(
            sdf_points,
            surface.cells,
            sdf_grid_coords,
            use_sign_winding_number=self.use_winding_number,
        )

        new_gd = domain.global_data.clone()
        new_gd[self.grid_field] = stored_grid
        new_gd[self.sdf_field] = sdf_grid

        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        return (
            f"boundary={self.boundary_name!r}, "
            f"grid_resolution={self.grid_resolution}, "
            f"grid_field={self.grid_field!r}, sdf_field={self.sdf_field!r}, "
            f"normalize_coords={self.normalize_coords}"
        )


# --------------------------------------------------------------------------- #
# SurfaceKNNNeighbors
# --------------------------------------------------------------------------- #


@register()
class SurfaceKNNNeighbors(MeshTransform):
    r"""Compute per-cell k-nearest-neighbour information on a surface mesh.

    For each triangle centroid the transform finds the *k* nearest other
    centroids (excluding self) and stores their centroids, normals, and
    areas as extra ``cell_data`` fields on the target boundary:

    - ``{neighbors_field}``          : Tensor :math:`(N_c, k-1, 3)`
    - ``{neighbor_normals_field}``   : Tensor :math:`(N_c, k-1, 3)`
    - ``{neighbor_areas_field}``     : Tensor :math:`(N_c, k-1)`

    The ``k-1`` comes from DoMINO's convention of dropping the self-neighbour
    (the closest hit is the query point itself, distance zero).

    Uses :func:`physicsnemo.nn.functional.knn`, which auto-dispatches to the
    cuml implementation on CUDA (via cupy dlpack, zero-copy), scipy on CPU,
    or a torch fallback.  No new kernel work.

    Parameters
    ----------
    k : int
        Number of neighbours to return (including self).  DoMINO's default
        is 11 (10 real neighbours + self).
    boundary_name : str
        Name of the :class:`DomainMesh` boundary to operate on.
    neighbors_field, neighbor_normals_field, neighbor_areas_field : str
        ``cell_data`` key names for the three output tensors.

    Notes
    -----
    - If the boundary has fewer than ``k`` cells the transform pads with
      the available cells -- kNN returns indices in ``[0, n_cells)`` so
      after slicing, self-index filtering may be incomplete.  This is
      consistent with the legacy pipeline which assumes
      ``n_cells >= surface_points_sample + 1``.
    """

    def __init__(
        self,
        *,
        k: int = 11,
        boundary_name: str = "boundary",
        neighbors_field: str = "surface_mesh_neighbors",
        neighbor_normals_field: str = "surface_neighbors_normals",
        neighbor_areas_field: str = "surface_neighbors_areas",
    ) -> None:
        super().__init__()
        if k < 2:
            raise ValueError(f"k must be >= 2 (1 real neighbour + self); got {k}")
        self.k = int(k)
        self.boundary_name = boundary_name
        self.neighbors_field = neighbors_field
        self.neighbor_normals_field = neighbor_normals_field
        self.neighbor_areas_field = neighbor_areas_field

    def _compute_neighbors_on_mesh(self, mesh: Mesh) -> Mesh:
        """Shared body: compute kNN on a single surface mesh and append cell_data."""
        centroids = mesh.cell_centroids.float()  # (N_c, 3)
        normals = mesh.cell_normals.float()  # (N_c, 3)
        areas = mesh.cell_areas.float()  # (N_c,)

        indices, _ = knn(centroids, centroids, self.k)
        # Drop self (the closest hit is the query itself, at distance 0).
        # Legacy: ``full_surface_coordinates[neighbor_indices][:, 1:]``.
        neighbor_indices = indices[:, 1:]  # (N_c, k-1)

        neighbors = centroids[neighbor_indices]  # (N_c, k-1, 3)
        neighbor_normals = normals[neighbor_indices]  # (N_c, k-1, 3)
        neighbor_areas = areas[neighbor_indices]  # (N_c, k-1)

        new_cd = mesh.cell_data.clone()
        new_cd[self.neighbors_field] = neighbors
        new_cd[self.neighbor_normals_field] = neighbor_normals
        new_cd[self.neighbor_areas_field] = neighbor_areas

        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=new_cd,
            global_data=mesh.global_data,
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        return self._compute_neighbors_on_mesh(mesh)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        if self.boundary_name not in domain.boundaries:
            raise KeyError(
                f"Boundary {self.boundary_name!r} not found. "
                f"Available: {domain.boundary_names}"
            )
        new_boundaries = dict(domain.boundaries)
        new_boundaries[self.boundary_name] = self._compute_neighbors_on_mesh(
            domain.boundaries[self.boundary_name]
        )
        return DomainMesh(
            interior=domain.interior,
            boundaries=new_boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"k={self.k}, boundary={self.boundary_name!r}, "
            f"neighbors_field={self.neighbors_field!r}"
        )


# --------------------------------------------------------------------------- #
# ComputeDoMINOPositionalEncodings
# --------------------------------------------------------------------------- #


@register()
class ComputeDoMINOPositionalEncodings(MeshTransform):
    r"""Emit DoMINO's three positional-encoding fields.

    Legacy ``DoMINODataPipe`` materializes:

    - ``pos_surface_center_of_mass``  := ``surface_coords - com``,
    - ``pos_volume_center_of_mass``   := ``volume_coords  - com``,
    - ``pos_volume_closest``          := ``volume_coords  - closest_surface_point``.

    ``com`` is the area-weighted centre of mass of the surface geometry
    (from :func:`physicsnemo.models.domino.utils.calculate_center_of_mass`,
    same as ``CenterMesh(use_area_weighting=True)`` uses), **computed once
    from world-coord STL centroids / areas**.  Legacy then normalizes COM
    and coords into :math:`[-1, 1]` *just before* each subtraction --
    independently per frame (surface bbox for the surface branch, volume
    bbox for the volume branch); see
    ``physicsnemo/datapipes/cae/domino_datapipe.py`` at git sha
    ``d2eb6cc8``, lines 493-499 (surface) and 589-598 / 620-635 (volume).

    To reproduce that behaviour in the PR #1512 YAML pipeline -- where
    :class:`NormalizeDomainByBBox` runs *after* this transform to rescale
    the bulk geometry -- this transform accepts optional per-frame bbox
    kwargs and normalizes its own inputs inline.  When the kwargs are
    supplied, the emitted ``pos_*`` tensors live in the normalized frame
    (matching legacy and matching what DoMINO's forward pass consumes
    downstream).  When they are left ``None``, the subtraction happens in
    world coords -- back-compat with prior recipes.

    This transform must run **after**
    :class:`~sdf.ComputeSDFFromBoundary`.  When SDF closest points have
    already been written to
    ``interior.point_data[sdf_closest_points_field]`` -- in the same
    coordinate frame as the volume points (world if ``volume_bbox_*``
    is unset, normalized otherwise) -- this transform reuses them
    directly and avoids a second BVH pass.  Otherwise it recomputes the
    closest-point query via
    :func:`physicsnemo.nn.functional.signed_distance_field` on the
    target boundary.  The recompute path is a correctness-preserving
    fallback for configs that have not wired the reuse contract; when
    ``volume_bbox_*`` is set the recompute queries SDF on a normalized
    STL against normalized volume points, mirroring legacy
    ``process_volume`` where both inputs are normalized before
    ``signed_distance_field`` runs.

    Writes results into the appropriate data section:

    - ``interior.point_data[pos_volume_closest_field]`` :math:`(N_v, 3)`
    - ``interior.point_data[pos_volume_com_field]`` :math:`(N_v, 3)`
    - ``target_boundary.cell_data[pos_surface_com_field]`` :math:`(N_s, 3)`

    Parameters
    ----------
    stl_boundary : str
        Name of the boundary used as the STL geometry / area-weighting
        source.  Cell centroids and cell areas from this boundary feed
        ``calculate_center_of_mass``.
    target_boundary : str
        Name of the boundary whose cell centroids receive the surface
        positional encoding (``pos_surface_center_of_mass``).  Often the
        same as *stl_boundary* -- in that case a single centroid list is
        used for both operations.
    pos_surface_com_field, pos_volume_closest_field, pos_volume_com_field : str
        ``cell_data`` / ``point_data`` keys for the three output fields.
    sdf_closest_points_field : str or None
        If set, read precomputed closest points from
        ``interior.point_data[sdf_closest_points_field]`` instead of
        recomputing via ``signed_distance_field``.  Expected shape
        :math:`(N_v, 3)`.  When ``volume_bbox_*`` is also set, the
        precomputed tensor is assumed to live in the same frame as the
        volume coords used for subtraction (callers are responsible for
        this invariant if they mix normalization upstream).
    surface_bbox_min, surface_bbox_max : sequence of 3 floats or None
        Axis-aligned surface-frame bbox corners.  When provided (and
        ``compute_surface_encodings=True``) the target cell centroids and
        COM are rescaled into :math:`[-1, 1]` by this bbox before the
        subtraction that emits ``pos_surface_center_of_mass``.
        Default ``None`` (no internal normalization).
    volume_bbox_min, volume_bbox_max : sequence of 3 floats or None
        Axis-aligned volume-frame bbox corners.  When provided (and
        ``compute_volume_encodings=True``) the interior points, the STL
        vertices, and the COM are rescaled into :math:`[-1, 1]` by this
        bbox before the subtractions that emit
        ``pos_volume_center_of_mass`` and ``pos_volume_closest``.
        Default ``None`` (no internal normalization).
    """

    def __init__(
        self,
        *,
        stl_boundary: str = "stl_geometry",
        target_boundary: str = "boundary",
        pos_surface_com_field: str = "pos_surface_center_of_mass",
        pos_volume_closest_field: str = "pos_volume_closest",
        pos_volume_com_field: str = "pos_volume_center_of_mass",
        sdf_closest_points_field: str | None = None,
        compute_volume_encodings: bool = True,
        compute_surface_encodings: bool = True,
        surface_bbox_min: Sequence[float] | None = None,
        surface_bbox_max: Sequence[float] | None = None,
        volume_bbox_min: Sequence[float] | None = None,
        volume_bbox_max: Sequence[float] | None = None,
        surface_target_section: Literal["cell_data", "point_data"] = "cell_data",
    ) -> None:
        super().__init__()
        self.stl_boundary = stl_boundary
        self.target_boundary = target_boundary
        self.pos_surface_com_field = pos_surface_com_field
        self.pos_volume_closest_field = pos_volume_closest_field
        self.pos_volume_com_field = pos_volume_com_field
        self.sdf_closest_points_field = sdf_closest_points_field
        self.compute_volume_encodings = compute_volume_encodings
        self.compute_surface_encodings = compute_surface_encodings
        if surface_target_section not in ("cell_data", "point_data"):
            raise ValueError(
                "surface_target_section must be 'cell_data' or 'point_data', "
                f"got {surface_target_section!r}."
            )
        self.surface_target_section = surface_target_section

        # Validate bbox pairs together (both or neither per frame).  Leave
        # self._surface_bbox / self._volume_bbox as None when unset so the
        # body can cheap-check ``is not None`` to gate the normalize step.
        self._surface_bbox: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        )
        if surface_bbox_min is None and surface_bbox_max is None:
            self._surface_bbox = None
        elif surface_bbox_min is not None and surface_bbox_max is not None:
            self._surface_bbox = _validate_bbox(surface_bbox_min, surface_bbox_max)
        else:
            raise ValueError(
                "surface_bbox_min and surface_bbox_max must be provided "
                "together or both left as None."
            )
        self._volume_bbox: (
            tuple[tuple[float, float, float], tuple[float, float, float]] | None
        )
        if volume_bbox_min is None and volume_bbox_max is None:
            self._volume_bbox = None
        elif volume_bbox_min is not None and volume_bbox_max is not None:
            self._volume_bbox = _validate_bbox(volume_bbox_min, volume_bbox_max)
        else:
            raise ValueError(
                "volume_bbox_min and volume_bbox_max must be provided "
                "together or both left as None."
            )

    def __call__(self, mesh: Mesh) -> Mesh:
        # Single Mesh is meaningful only in degenerate cases; we require a
        # DomainMesh so we know which boundary is "STL" vs "target".
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        if self.stl_boundary not in domain.boundaries:
            raise KeyError(
                f"STL boundary {self.stl_boundary!r} not found. "
                f"Available: {domain.boundary_names}"
            )
        stl = domain.boundaries[self.stl_boundary]

        # Center of mass from the STL surface (cell centroids area-weighted),
        # in world coords.  Legacy computes this once (domino_datapipe.py
        # l737-739 @ d2eb6cc8) and re-normalizes it per-frame below.
        com_world = calculate_center_of_mass(
            stl.cell_centroids.float(),
            stl.cell_areas.float(),
        )  # (1, 3)

        new_boundaries = dict(domain.boundaries)
        new_interior = domain.interior

        # --- Surface-side encoding: pos_surface_center_of_mass ---
        # Legacy process_surface l493-499: normalize surface coords and COM
        # by the surface bbox, then subtract.  In the back-compat branch
        # (surface_bbox is None) both operands stay in world coords.
        if self.compute_surface_encodings:
            if self.target_boundary not in domain.boundaries:
                raise KeyError(
                    f"Target boundary {self.target_boundary!r} not found. "
                    f"Available: {domain.boundary_names}"
                )
            target = domain.boundaries[self.target_boundary]
            # Point-mode emits one encoding per vertex (target.points); cell
            # mode emits one per face (target.cell_centroids).  Downstream
            # ``surface_mesh_centers`` must be chosen from the same section
            # so encoding shapes line up with DoMINO's forward pass.
            if self.surface_target_section == "point_data":
                target_coords = target.points.float()
            else:
                target_coords = target.cell_centroids.float()
            if self._surface_bbox is not None:
                target_coords_n = _rescale_bbox(target_coords, self._surface_bbox)
                com_s_n = _rescale_bbox(com_world, self._surface_bbox)
            else:
                target_coords_n = target_coords
                com_s_n = com_world
            pos_surface_com = target_coords_n - com_s_n  # (N_s, 3)

            if self.surface_target_section == "point_data":
                new_target_pd = target.point_data.clone()
                new_target_pd[self.pos_surface_com_field] = pos_surface_com
                new_boundaries[self.target_boundary] = Mesh(
                    points=target.points,
                    cells=target.cells,
                    point_data=new_target_pd,
                    cell_data=target.cell_data,
                    global_data=target.global_data,
                )
            else:
                new_target_cd = target.cell_data.clone()
                new_target_cd[self.pos_surface_com_field] = pos_surface_com
                new_boundaries[self.target_boundary] = Mesh(
                    points=target.points,
                    cells=target.cells,
                    point_data=target.point_data,
                    cell_data=new_target_cd,
                    global_data=target.global_data,
                )

        # --- Volume-side encodings: pos_volume_center_of_mass + pos_volume_closest ---
        # Legacy process_volume l589-598 normalizes volume coords, STL
        # vertices, and COM by the volume bbox; l620-635 then runs
        # signed_distance_field on the normalized pair and subtracts.
        if self.compute_volume_encodings:
            volume_points = domain.interior.points.float()
            if self._volume_bbox is not None:
                volume_points_n = _rescale_bbox(volume_points, self._volume_bbox)
                com_v_n = _rescale_bbox(com_world, self._volume_bbox)
            else:
                volume_points_n = volume_points
                com_v_n = com_world

            pos_volume_com = volume_points_n - com_v_n

            if (
                self.sdf_closest_points_field is not None
                and self.sdf_closest_points_field in domain.interior.point_data.keys()
            ):
                # Caller promises this tensor lives in the same frame as
                # volume_points_n (either both world or both normalized).
                closest_points = domain.interior.point_data[
                    self.sdf_closest_points_field
                ].float()
            else:
                # Fallback: compute closest points via SDF on the STL boundary.
                # When volume_bbox is set, run SDF in the normalized frame so
                # the returned closest-point is already normalized, matching
                # legacy process_volume's inline normalize-then-SDF pattern.
                if self._volume_bbox is not None:
                    stl_vertices_sdf = _rescale_bbox(
                        stl.points.float(), self._volume_bbox
                    )
                    query_points_sdf = volume_points_n
                else:
                    stl_vertices_sdf = stl.points.float()
                    query_points_sdf = volume_points
                _, closest_points = signed_distance_field(
                    stl_vertices_sdf,
                    stl.cells,
                    query_points_sdf,
                    use_sign_winding_number=True,
                )

            pos_volume_closest = volume_points_n - closest_points  # (N_v, 3)

            new_interior_pd = domain.interior.point_data.clone()
            new_interior_pd[self.pos_volume_com_field] = pos_volume_com
            new_interior_pd[self.pos_volume_closest_field] = pos_volume_closest

            new_interior = Mesh(
                points=domain.interior.points,
                cells=domain.interior.cells,
                point_data=new_interior_pd,
                cell_data=domain.interior.cell_data,
                global_data=domain.interior.global_data,
            )

        return DomainMesh(
            interior=new_interior,
            boundaries=new_boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"stl={self.stl_boundary!r}, target={self.target_boundary!r}, "
            f"volume={self.compute_volume_encodings}, "
            f"surface={self.compute_surface_encodings}, "
            f"surface_bbox={self._surface_bbox}, "
            f"volume_bbox={self._volume_bbox}"
        )


# --------------------------------------------------------------------------- #
# CropMeshToBBox
# --------------------------------------------------------------------------- #


@register()
class CropMeshToBBox(MeshTransform):
    r"""Drop interior points (or boundary points) outside an axis-aligned bbox.

    Equivalent to legacy ``sample_in_bbox=True`` on a Mesh object.
    Applies ``min < x < max`` strict inequalities per axis (matching
    legacy behaviour in ``DoMINODataPipe.process_volume``).

    Uses :meth:`Mesh.slice_points`, which automatically drops cells that
    reference removed points and remaps cell indices.

    Parameters
    ----------
    bbox_min, bbox_max : sequence of 3 floats
        Axis-aligned bounding box corners.  Points outside are removed.
    target : {"interior", "boundary"}
        Which part of the :class:`DomainMesh` to crop.  ``"interior"`` is
        the usual DoMINO volume-bbox filter.  ``"boundary"`` crops a named
        boundary (specified via *boundary_name*).
    boundary_name : str
        Required when ``target == "boundary"``.  Ignored otherwise.
    """

    def __init__(
        self,
        *,
        bbox_min: Sequence[float],
        bbox_max: Sequence[float],
        target: Literal["interior", "boundary"] = "interior",
        boundary_name: str = "boundary",
    ) -> None:
        super().__init__()
        self.bbox_min, self.bbox_max = _validate_bbox(bbox_min, bbox_max)
        self.target = target
        self.boundary_name = boundary_name

    def _crop(self, mesh: Mesh) -> Mesh:
        device = mesh.points.device
        mn = torch.tensor(self.bbox_min, dtype=mesh.points.dtype, device=device)
        mx = torch.tensor(self.bbox_max, dtype=mesh.points.dtype, device=device)
        inside = ((mesh.points > mn) & (mesh.points < mx)).all(dim=-1)
        return mesh.slice_points(inside)

    def __call__(self, mesh: Mesh) -> Mesh:
        return self._crop(mesh)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        if self.target == "interior":
            new_interior = self._crop(domain.interior)
            return DomainMesh(
                interior=new_interior,
                boundaries=domain.boundaries,
                global_data=domain.global_data,
            )
        if self.target == "boundary":
            if self.boundary_name not in domain.boundaries:
                raise KeyError(
                    f"Boundary {self.boundary_name!r} not found. "
                    f"Available: {domain.boundary_names}"
                )
            new_boundaries = dict(domain.boundaries)
            new_boundaries[self.boundary_name] = self._crop(
                domain.boundaries[self.boundary_name]
            )
            return DomainMesh(
                interior=domain.interior,
                boundaries=new_boundaries,
                global_data=domain.global_data,
            )
        raise ValueError(f"Unknown crop target {self.target!r}")

    def extra_repr(self) -> str:
        return f"bbox=[{self.bbox_min},{self.bbox_max}], target={self.target!r}"


# --------------------------------------------------------------------------- #
# SubsampleNamedBoundary
# --------------------------------------------------------------------------- #


@register()
class SubsampleNamedBoundary(MeshTransform):
    r"""Randomly subsample the vertices of a named :class:`DomainMesh` boundary.

    Wraps :func:`physicsnemo.datapipes.transforms.subsample.shuffle_array` to
    pick ``n_points`` vertices without replacement from
    ``domain.boundaries[boundary_name].points``, then rebuilds the boundary
    mesh via :meth:`Mesh.slice_points` so that ``point_data`` is kept in
    correspondence and connectivity is reduced to the (usually empty) set of
    cells whose vertices all survived.

    Legacy equivalent: DoMINO's ``geom_points_sample`` knob (default
    ``200_000``) applied to the STL vertex list produced by
    ``DoMINODataPipe.downsample_geometry`` (``physicsnemo/datapipes/cae/
    domino_datapipe.py`` @ ``d2eb6cc8``).  The output is consumed as
    ``geometry_coordinates`` by :class:`DoMINO.forward`.

    Parameters
    ----------
    boundary_name : str
        Name of the :class:`DomainMesh` boundary to subsample.  Typically
        ``"stl_geometry"`` for DoMINO.
    n_points : int
        Target number of vertices to keep.  If the boundary has fewer
        than ``n_points`` vertices, all of them are returned unchanged
        (matching ``shuffle_array`` semantics).  Default ``200_000``.

    Notes
    -----
    - The resulting boundary retains only those cells whose three
      vertices all landed in the random subset.  In practice, for
      random downsampling of a large STL (:math:`> 10^6` vertices) to
      :math:`2 \cdot 10^5`, essentially zero cells survive -- which is
      exactly what the legacy pipeline assumed when it discarded
      connectivity and kept only the point cloud.
    - The sampling is non-deterministic across calls.  Seed via
      :func:`torch.manual_seed` upstream if reproducibility is needed.
    """

    def __init__(
        self,
        *,
        boundary_name: str = "stl_geometry",
        n_points: int = 200_000,
    ) -> None:
        super().__init__()
        if n_points <= 0:
            raise ValueError(f"n_points must be positive; got {n_points}")
        self.boundary_name = boundary_name
        self.n_points = int(n_points)

    def __call__(self, mesh: Mesh) -> Mesh:
        # Single-Mesh path: subsample the mesh's own points.  slice_points
        # handles point_data alignment and cell remapping automatically.
        if mesh.points.shape[0] <= self.n_points:
            return mesh
        _, indices = shuffle_array(mesh.points, self.n_points)
        return mesh.slice_points(indices)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        if self.boundary_name not in domain.boundaries:
            raise KeyError(
                f"Boundary {self.boundary_name!r} not found. "
                f"Available: {domain.boundary_names}"
            )
        boundary = domain.boundaries[self.boundary_name]
        if boundary.points.shape[0] <= self.n_points:
            return domain

        _, indices = shuffle_array(boundary.points, self.n_points)
        new_boundaries = dict(domain.boundaries)
        new_boundaries[self.boundary_name] = boundary.slice_points(indices)
        return DomainMesh(
            interior=domain.interior,
            boundaries=new_boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return f"boundary={self.boundary_name!r}, n_points={self.n_points}"


# --------------------------------------------------------------------------- #
# LiftCellCentroidsAndAreas
# --------------------------------------------------------------------------- #


@register()
class LiftCellCentroidsAndAreas(MeshTransform):
    r"""Materialize cell centroids and cell areas as explicit ``cell_data`` fields.

    :class:`Mesh` exposes ``cell_centroids`` and ``cell_areas`` as computed
    properties derived from ``points`` and ``cells``. DoMINO's ``data_dict``
    contract requires the centroids and areas to be reachable via DomainMesh
    paths, so this transform lifts them into explicit ``cell_data`` entries
    before the recipe collate resolves ``forward_kwargs``.

    Parameters
    ----------
    boundary_name : str
        Name of the :class:`DomainMesh` boundary on which to lift the
        fields.  Defaults to ``"boundary"``, matching legacy DoMINO's
        target-surface convention.
    centroids_field, areas_field : str
        ``cell_data`` keys under which to store the lifted tensors.
        Defaults mirror the names DoMINO's input dict expects.

    Notes
    -----
    Both tensors are float32 (matching ``Mesh.cell_centroids`` /
    ``cell_areas`` dtypes).  Shapes are :math:`(N_c, 3)` for centroids and
    :math:`(N_c,)` for areas, where :math:`N_c` is the number of cells.
    """

    def __init__(
        self,
        *,
        boundary_name: str = "boundary",
        centroids_field: str = "cell_centroids",
        areas_field: str = "cell_areas",
    ) -> None:
        super().__init__()
        self.boundary_name = boundary_name
        self.centroids_field = centroids_field
        self.areas_field = areas_field

    def _lift(self, mesh: Mesh) -> Mesh:
        new_cd = mesh.cell_data.clone()
        new_cd[self.centroids_field] = mesh.cell_centroids.float()
        new_cd[self.areas_field] = mesh.cell_areas.float()
        return Mesh(
            points=mesh.points,
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=new_cd,
            global_data=mesh.global_data,
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        return self._lift(mesh)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        if self.boundary_name not in domain.boundaries:
            raise KeyError(
                f"Boundary {self.boundary_name!r} not found. "
                f"Available: {domain.boundary_names}"
            )
        new_boundaries = dict(domain.boundaries)
        new_boundaries[self.boundary_name] = self._lift(
            domain.boundaries[self.boundary_name]
        )
        return DomainMesh(
            interior=domain.interior,
            boundaries=new_boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"boundary={self.boundary_name!r}, "
            f"centroids_field={self.centroids_field!r}, "
            f"areas_field={self.areas_field!r}"
        )


# --------------------------------------------------------------------------- #
# PromoteBoundaryToInterior
# --------------------------------------------------------------------------- #


@register()
class PromoteBoundaryToInterior(MeshTransform):
    r"""Expose boundary cell data as the DomainMesh interior point cloud.

    The unified recipe's trainer extracts targets from
    ``interior.point_data``. Surface-only DoMINO predicts at surface cell
    centers, so this transform promotes a named boundary's lifted
    ``cell_centroids`` into ``interior.points`` and copies selected
    ``cell_data`` fields into the new interior ``point_data``.

    Args:
        boundary_name: Boundary to promote.
        centroids_field: Boundary ``cell_data`` key containing cell centers.
        target_fields: Boundary ``cell_data`` keys to copy as training
            targets under the same names.
        copy_cell_data: Mapping of ``{dest_name: source_name}`` for
            auxiliary DoMINO inputs to copy from boundary ``cell_data`` into
            ``interior.point_data``.
    """

    def __init__(
        self,
        *,
        boundary_name: str = "boundary",
        centroids_field: str = "cell_centroids",
        target_fields: Sequence[str] = (),
        copy_cell_data: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.boundary_name = boundary_name
        self.centroids_field = centroids_field
        self.target_fields = tuple(target_fields)
        self.copy_cell_data = dict(copy_cell_data or {})

    def _promote_mesh(self, mesh: Mesh) -> Mesh:
        if self.centroids_field not in mesh.cell_data.keys():
            raise KeyError(
                f"Cell-data field {self.centroids_field!r} not found on "
                f"boundary {self.boundary_name!r}."
            )

        point_data: dict[str, torch.Tensor] = {}
        for name in self.target_fields:
            if name not in mesh.cell_data.keys():
                raise KeyError(
                    f"Target field {name!r} not found in "
                    f"{self.boundary_name!r}.cell_data."
                )
            point_data[name] = mesh.cell_data[name]

        for dest, source in self.copy_cell_data.items():
            if source not in mesh.cell_data.keys():
                raise KeyError(
                    f"Auxiliary field {source!r} not found in "
                    f"{self.boundary_name!r}.cell_data."
                )
            value = mesh.cell_data[source]
            if value.ndim == 1:
                value = value.unsqueeze(-1)
            point_data[dest] = value

        return Mesh(
            points=mesh.cell_data[self.centroids_field],
            point_data=point_data,
            global_data=mesh.global_data,
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        return self._promote_mesh(mesh)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        if self.boundary_name not in domain.boundaries:
            raise KeyError(
                f"Boundary {self.boundary_name!r} not found. "
                f"Available: {domain.boundary_names}"
            )
        return DomainMesh(
            interior=self._promote_mesh(domain.boundaries[self.boundary_name]),
            boundaries=domain.boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return (
            f"boundary={self.boundary_name!r}, "
            f"centroids_field={self.centroids_field!r}, "
            f"target_fields={self.target_fields!r}"
        )


# --------------------------------------------------------------------------- #
# NormalizeDomainByBBox
# --------------------------------------------------------------------------- #


@register()
class NormalizeDomainByBBox(MeshTransform):
    r"""Rescale every spatial tensor in a :class:`DomainMesh` into :math:`[-1, 1]`.

    Applies the legacy DoMINO normalization
    :math:`\mathrm{norm}(x, \min, \max) = 2 (x - \min) / (\max - \min) - 1`
    element-wise per axis (see ``physicsnemo.models.domino.utils.utils.normalize``
    and ``physicsnemo/datapipes/cae/domino_datapipe.py`` at git sha
    ``d2eb6cc8``) to every tensor that lives in world coordinates after the
    earlier pipeline stages have run.

    Unlike :class:`NormalizeMeshFields` (which rescales *data* fields like
    pressure/velocity), this transform rescales *geometry*: point clouds,
    cell centroids, and latent-grid coordinates. It is typically the last
    geometry-touching transform before the recipe collate resolves
    ``forward_kwargs``.

    The transform supports two distinct bboxes -- a ``surface_bbox`` (used
    for the surface geometry rep's ``surf_grid`` + surface boundary points)
    and a ``volume_bbox`` (used for the volume geometry rep's ``grid`` +
    interior points) -- matching the legacy HLPW/DoMINO convention of
    allowing a tighter surface box nested inside the volumetric one.  For
    datasets where the two coincide (the HLPW default), pass the same
    corners to both.

    Parameters
    ----------
    surface_bbox_min, surface_bbox_max : sequence of 3 floats
        Axis-aligned bbox corners for surface-frame normalization.  Applied
        to the surface boundary points, the STL boundary points, the surface
        boundary's ``cell_centroids`` cell-data field, and the global grids
        listed in ``surface_grid_fields``.  Also emitted as a ``(2, 3)``
        tensor under ``global_data[{surface_min_max_field}]``.
    volume_bbox_min, volume_bbox_max : sequence of 3 floats
        Axis-aligned bbox corners for volume-frame normalization.  Applied
        to ``interior.points`` and to the global grids listed in
        ``volume_grid_fields``.  Emitted as ``(2, 3)`` under
        ``global_data[{volume_min_max_field}]``.
    normalize_interior : bool
        If ``True``, rescale ``interior.points`` with the volume bbox.
        Default ``True``.
    surface_boundaries : sequence of str
        Names of boundaries whose ``points`` (and whose
        ``cell_data.cell_centroids`` if present) are rescaled with the
        surface bbox.  Typically ``("boundary",)`` plus ``"stl_geometry"``
        when the STL is still part of the domain.  Default ``()``.
    surface_grid_fields, volume_grid_fields : sequence of str
        Names of global-data tensors to rescale with the surface and volume
        bboxes respectively.  Fields not present in ``global_data`` are
        silently skipped (mirrors :class:`NormalizeMeshFields` behaviour).
        Defaults: ``("surf_grid",)`` and ``("grid",)``.
    centroids_field : str
        Name of the ``cell_data`` entry that :class:`LiftCellCentroidsAndAreas`
        produced; looked up on every surface boundary and rescaled when
        present.  Default ``"cell_centroids"``.
    neighbor_coord_fields : sequence of str
        Names of ``cell_data`` entries holding world-coordinate neighbor
        points (e.g. the ``(N_c, k-1, 3)`` tensor emitted by
        :class:`SurfaceKNNNeighbors`) that must be rescaled with the same
        bbox as ``centroids_field`` so downstream consumers see both in the
        same frame.  Matches legacy behaviour where ``surface_neighbors``
        is normalized alongside ``surface_coordinates`` in
        ``DoMINODataPipe.process_surface`` (git sha ``d2eb6cc8``).  Fields
        not present on a given boundary are skipped silently.  Default
        ``("surface_mesh_neighbors",)``.
    surface_min_max_field, volume_min_max_field : str
        Keys under which to emit the ``(2, 3)`` min/max tensors into
        ``global_data``.  Set to ``None`` to skip emission for that frame.
        Defaults: ``"surface_min_max"`` and ``"volume_min_max"``.

    Notes
    -----
    - Normalization of interior ``point_data`` is **not** performed -- those
      are physical fields (pressure, velocity), not coordinates.  Use
      :class:`NormalizeMeshFields` for them.
    - For HLPW where the surface and volume bboxes coincide, the two
      min/max tensors emitted are identical; downstream code can still key
      off either name.  This matches legacy behaviour.
    - This transform replaces the earlier ``NormalizeCoordinatesByBBox``.
    """

    def __init__(
        self,
        *,
        surface_bbox_min: Sequence[float],
        surface_bbox_max: Sequence[float],
        volume_bbox_min: Sequence[float],
        volume_bbox_max: Sequence[float],
        normalize_interior: bool = True,
        surface_boundaries: Sequence[str] = (),
        surface_grid_fields: Sequence[str] = ("surf_grid",),
        volume_grid_fields: Sequence[str] = ("grid",),
        centroids_field: str = "cell_centroids",
        neighbor_coord_fields: Sequence[str] = ("surface_mesh_neighbors",),
        surface_min_max_field: str | None = "surface_min_max",
        volume_min_max_field: str | None = "volume_min_max",
    ) -> None:
        super().__init__()
        self._surface_bbox = _validate_bbox(surface_bbox_min, surface_bbox_max)
        self._volume_bbox = _validate_bbox(volume_bbox_min, volume_bbox_max)
        self.normalize_interior = bool(normalize_interior)
        self.surface_boundaries = tuple(surface_boundaries)
        self.surface_grid_fields = tuple(surface_grid_fields)
        self.volume_grid_fields = tuple(volume_grid_fields)
        self.centroids_field = centroids_field
        self.neighbor_coord_fields = tuple(neighbor_coord_fields)
        self.surface_min_max_field = surface_min_max_field
        self.volume_min_max_field = volume_min_max_field

    @staticmethod
    def _rescale(
        x: Float[torch.Tensor, "..."],
        bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> Float[torch.Tensor, "..."]:
        """Apply :math:`2(x - \\min) / (\\max - \\min) - 1` broadcasting over trailing dim 3.

        Thin wrapper over :func:`_rescale_bbox` so callers that previously
        referenced ``NormalizeDomainByBBox._rescale`` keep working.
        """
        return _rescale_bbox(x, bbox)

    def _rescale_mesh_points(
        self,
        mesh: Mesh,
        bbox: tuple[tuple[float, float, float], tuple[float, float, float]],
    ) -> Mesh:
        """Return a new :class:`Mesh` with ``points`` and coordinate-valued
        ``cell_data`` tensors rescaled.

        Rescales ``mesh.points`` plus every ``cell_data`` entry named in
        ``[centroids_field, *neighbor_coord_fields]`` that is actually
        present on the mesh.  Missing fields are skipped silently so the
        same transform works for boundaries that have not run
        :class:`LiftCellCentroidsAndAreas` or :class:`SurfaceKNNNeighbors`.
        This keeps neighbor coordinates in the same frame as centroids,
        matching ``DoMINODataPipe.process_surface`` (git sha ``d2eb6cc8``).
        """
        coord_fields = (self.centroids_field, *self.neighbor_coord_fields)
        new_cd = mesh.cell_data
        cloned = False
        for name in coord_fields:
            if name in mesh.cell_data.keys():
                if not cloned:
                    new_cd = mesh.cell_data.clone()
                    cloned = True
                new_cd[name] = self._rescale(mesh.cell_data[name].float(), bbox)
        return Mesh(
            points=self._rescale(mesh.points, bbox),
            cells=mesh.cells,
            point_data=mesh.point_data,
            cell_data=new_cd,
            global_data=mesh.global_data,
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        # On a bare Mesh we only have points (no surface/volume distinction);
        # default to the volume bbox for consistency with most
        # DoMINO use-cases where the plain-Mesh path is the interior.
        return self._rescale_mesh_points(mesh, self._volume_bbox)

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        new_interior = domain.interior
        if self.normalize_interior:
            new_interior = self._rescale_mesh_points(domain.interior, self._volume_bbox)

        new_boundaries = dict(domain.boundaries)
        for name in self.surface_boundaries:
            if name not in domain.boundaries:
                raise KeyError(
                    f"Boundary {name!r} not found. Available: {domain.boundary_names}"
                )
            new_boundaries[name] = self._rescale_mesh_points(
                domain.boundaries[name], self._surface_bbox
            )

        new_gd = domain.global_data.clone()
        for key in self.surface_grid_fields:
            if key in new_gd.keys():
                new_gd[key] = self._rescale(new_gd[key].float(), self._surface_bbox)
        for key in self.volume_grid_fields:
            if key in new_gd.keys():
                new_gd[key] = self._rescale(new_gd[key].float(), self._volume_bbox)

        # Emit (2, 3) min/max tensors on the same device/dtype as the interior
        # points so downstream collate stacks work without extra .to() calls.
        ref = domain.interior.points
        if self.surface_min_max_field is not None:
            new_gd[self.surface_min_max_field] = torch.stack(
                [
                    torch.tensor(
                        self._surface_bbox[0], dtype=ref.dtype, device=ref.device
                    ),
                    torch.tensor(
                        self._surface_bbox[1], dtype=ref.dtype, device=ref.device
                    ),
                ]
            )
        if self.volume_min_max_field is not None:
            new_gd[self.volume_min_max_field] = torch.stack(
                [
                    torch.tensor(
                        self._volume_bbox[0], dtype=ref.dtype, device=ref.device
                    ),
                    torch.tensor(
                        self._volume_bbox[1], dtype=ref.dtype, device=ref.device
                    ),
                ]
            )

        return DomainMesh(
            interior=new_interior,
            boundaries=new_boundaries,
            global_data=new_gd,
        )

    def extra_repr(self) -> str:
        return (
            f"surface_bbox={self._surface_bbox}, volume_bbox={self._volume_bbox}, "
            f"surface_boundaries={self.surface_boundaries}"
        )


__all__ = [
    "ComputeGridSDFFromBoundary",
    "SurfaceKNNNeighbors",
    "ComputeDoMINOPositionalEncodings",
    "CropMeshToBBox",
    "SubsampleNamedBoundary",
    "LiftCellCentroidsAndAreas",
    "PromoteBoundaryToInterior",
    "NormalizeDomainByBBox",
]
