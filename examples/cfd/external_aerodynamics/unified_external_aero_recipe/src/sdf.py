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
SDF (Signed Distance Field) pipeline transforms for volume meshes.

Provides a transform that computes SDF + normals from a boundary surface
onto interior volume points, and a cleanup transform to drop temporary
boundaries before TensorDict conversion.

These work with ``DomainMeshReader``'s ``extra_boundaries`` parameter,
which loads a sibling STL mesh at full resolution alongside the domain
mesh.  The SDF transform reads the injected boundary, computes the
signed distance field, and writes results into ``interior.point_data``.

Recipe-local module registered into the global datapipe component
registry so components can be referenced via ``${dp:...}`` in Hydra
YAML configs.

Import this module before Hydra instantiation to register the components.
"""

from __future__ import annotations

from typing import Sequence
import torch

from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.nn.functional import signed_distance_field


@register()
class ComputeSDFFromBoundary(MeshTransform):
    r"""Compute SDF (and optionally normals + closest points) from a boundary surface.

    Reads the surface mesh from ``domain.boundaries[boundary_name]`` and
    evaluates the signed distance field at every interior point using
    :func:`physicsnemo.nn.functional.signed_distance_field` (Warp-backed,
    GPU-accelerated).

    Two operating frames
    --------------------
    By default the SDF is evaluated in **world coordinates** on the raw
    STL and raw interior points.  When both ``bbox_min`` and ``bbox_max``
    are supplied, the transform rescales the STL vertices and the interior
    query points into :math:`[-1, 1]` before the SDF call, so the resulting
    ``sdf_field`` values and ``closest_points_field`` vectors live in the
    **normalized frame**.  This mirrors legacy
    ``physicsnemo.datapipes.cae.domino_datapipe.py:614-630`` where both
    SDF calls are made on normalized STL / normalized queries, and it
    matches the frame expected by
    :class:`domino_transforms.ComputeDoMINOPositionalEncodings` when its
    own ``volume_bbox_*`` kwargs are set.

    Closest-points reuse
    --------------------
    When ``closest_points_field`` is set, the hit-point tensor returned
    by the Warp kernel is written to ``interior.point_data`` alongside
    the scalar SDF.  The downstream
    :class:`ComputeDoMINOPositionalEncodings` can then read that field
    via its ``sdf_closest_points_field`` kwarg and skip a second BVH
    pass.  The two transforms must agree on frame: either leave bbox
    unset on both and live in world coords, or set it on both and live
    in the normalized frame.

    Fields written to ``interior.point_data``:

    - ``{sdf_field}``                (N, 1) scalar SDF
    - ``{normals_field}``            (N, 3) unit normals, optional
    - ``{closest_points_field}``     (N, 3) closest hit points, optional

    Parameters
    ----------
    boundary_name : str
        Key of the boundary mesh to use as the SDF surface.
    sdf_field : str
        Name for the scalar SDF field in ``interior.point_data``.
    normals_field : str or None
        Optional name for the normals field.  ``None`` to skip.
        Cannot be combined with ``bbox_min``/``bbox_max`` (normals are
        only meaningful in the world frame; see ``apply_to_domain``).
    closest_points_field : str or None
        Optional name for the closest-hit-point field.  ``None`` to
        skip.  When supplied, the hit points already computed by the
        Warp kernel are persisted at no extra cost so downstream
        transforms can reuse them.
    use_winding_number : bool
        Whether to use winding-number sign computation.  Required for
        non-watertight meshes; slightly slower.
    bbox_min, bbox_max : sequence of 3 floats or None
        Optional bounding box.  When **both** are supplied, the STL and
        interior points are rescaled into :math:`[-1, 1]` before the
        SDF call, so the emitted ``sdf_field`` and
        ``closest_points_field`` live in the normalized frame (matching
        legacy ``process_volume``).  When both are ``None``, the SDF
        runs in world coordinates (back-compat).  Supplying only one is
        an error.
    """

    def __init__(
        self,
        boundary_name: str = "stl_geometry",
        sdf_field: str = "sdf",
        normals_field: str | None = "sdf_normals",
        closest_points_field: str | None = None,
        *,
        use_winding_number: bool = True,
        bbox_min: Sequence[float] | None = None,
        bbox_max: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        self.boundary_name = boundary_name
        self.sdf_field = sdf_field
        self.normals_field = normals_field
        self.closest_points_field = closest_points_field
        self.use_winding_number = use_winding_number

        if (bbox_min is None) != (bbox_max is None):
            raise ValueError(
                "bbox_min and bbox_max must both be provided or both be None"
            )
        self.bbox_min = (
            tuple(float(x) for x in bbox_min) if bbox_min is not None else None
        )
        self.bbox_max = (
            tuple(float(x) for x in bbox_max) if bbox_max is not None else None
        )
        if self.bbox_min is not None and len(self.bbox_min) != 3:
            raise ValueError(f"bbox_min must be length 3, got {self.bbox_min!r}")
        if self.bbox_max is not None and len(self.bbox_max) != 3:
            raise ValueError(f"bbox_max must be length 3, got {self.bbox_max!r}")
        if self.bbox_min is not None and self.normals_field is not None:
            raise ValueError(
                "ComputeSDFFromBoundary: normals_field cannot be combined "
                "with bbox_min/bbox_max.  Under non-uniform bbox rescaling "
                "the closest-point direction no longer corresponds to the "
                "world-frame surface normal.  Set normals_field=None when "
                "running in the normalized frame."
            )

    def __call__(self, mesh: Mesh) -> Mesh:
        # Single-mesh path is not meaningful for SDF (we need a separate
        # surface mesh).  Pass through unchanged.
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Compute SDF from the boundary surface onto interior points.

        Parameters
        ----------
        domain : DomainMesh
            Must contain a boundary named ``self.boundary_name`` with
            triangle cells.

        Returns
        -------
        DomainMesh
            Domain with SDF (and optionally normals + closest points)
            injected into ``interior.point_data``.
        """
        if self.boundary_name not in domain.boundaries:
            raise KeyError(
                f"Boundary {self.boundary_name!r} not found. "
                f"Available: {domain.boundary_names}"
            )

        surface = domain.boundaries[self.boundary_name]
        vertices = surface.points.float()
        faces = surface.cells

        if faces is None or faces.numel() == 0:
            raise ValueError(
                f"Boundary {self.boundary_name!r} has no cell connectivity "
                f"(required for SDF computation)"
            )

        query_points = domain.interior.points.float()

        if self.bbox_min is not None:
            # Normalized-frame SDF: rescale both STL and queries into
            # [-1, 1] before the BVH query so the emitted sdf / closest
            # points live in the normalized frame.  Mirrors legacy
            # process_volume at domino_datapipe.py:614-630 (d2eb6cc8).
            mn = torch.tensor(
                self.bbox_min, dtype=vertices.dtype, device=vertices.device
            )
            mx = torch.tensor(
                self.bbox_max, dtype=vertices.dtype, device=vertices.device
            )
            span = (mx - mn).clamp(min=1e-12)
            vertices_for_sdf = 2.0 * (vertices - mn) / span - 1.0
            query_points_for_sdf = 2.0 * (query_points - mn) / span - 1.0
        else:
            vertices_for_sdf = vertices
            query_points_for_sdf = query_points

        sdf_values, closest_points = signed_distance_field(
            vertices_for_sdf,
            faces,
            query_points_for_sdf,
            use_sign_winding_number=self.use_winding_number,
        )

        # Build updated point_data with SDF (N, 1)
        new_pd = domain.interior.point_data.clone()
        new_pd[self.sdf_field] = sdf_values.unsqueeze(-1)

        # Optionally persist the closest hit points.  Already returned
        # by the kernel; writing them back costs one reference.  Frame
        # matches sdf_values (world if bbox unset, normalized otherwise).
        if self.closest_points_field is not None:
            new_pd[self.closest_points_field] = closest_points

        # Optionally compute approximate normals from closest-point
        # direction.  Only reachable in the world-frame branch (guarded
        # in __init__), so query_points and closest_points share units.
        if self.normals_field is not None:
            normals = query_points - closest_points

            # Fallback for points on the surface (zero distance):
            # use direction from center of mass instead.
            dist = torch.norm(normals, dim=-1)
            on_surface = dist < 1e-6
            if on_surface.any():
                com = vertices.mean(dim=0, keepdim=True)
                normals[on_surface] = query_points[on_surface] - com

            # Normalize to unit vectors
            norm = torch.norm(normals, dim=-1, keepdim=True).clamp(min=1e-8)
            normals = normals / norm
            new_pd[self.normals_field] = normals

        new_interior = Mesh(
            points=domain.interior.points,
            cells=domain.interior.cells,
            point_data=new_pd,
            cell_data=domain.interior.cell_data,
            global_data=domain.interior.global_data,
        )

        return DomainMesh(
            interior=new_interior,
            boundaries=domain.boundaries,
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        parts = [
            f"boundary={self.boundary_name!r}",
            f"sdf_field={self.sdf_field!r}",
        ]
        if self.normals_field:
            parts.append(f"normals_field={self.normals_field!r}")
        if self.closest_points_field:
            parts.append(f"closest_points_field={self.closest_points_field!r}")
        if self.bbox_min is not None:
            parts.append(f"bbox_min={self.bbox_min}")
            parts.append(f"bbox_max={self.bbox_max}")
        parts.append(f"winding_number={self.use_winding_number}")
        return ", ".join(parts)


@register()
class DropBoundary(MeshTransform):
    r"""Remove one or more boundaries from a :class:`DomainMesh`.

    Useful for stripping temporary data (e.g. a full-resolution STL
    boundary injected for SDF computation) before downstream transforms
    like ``MeshToTensorDict`` that would otherwise serialize the large
    surface into the output TensorDict.

    Parameters
    ----------
    names : list[str]
        Boundary names to remove.
    """

    def __init__(self, names: list[str]) -> None:
        super().__init__()
        self.names = set(names)

    def __call__(self, mesh: Mesh) -> Mesh:
        # Single-mesh path: nothing to drop.
        return mesh

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Remove the named boundaries from the domain.

        Parameters
        ----------
        domain : DomainMesh
            Input domain mesh.

        Returns
        -------
        DomainMesh
            Domain mesh without the dropped boundaries.
        """
        return DomainMesh(
            interior=domain.interior,
            boundaries=domain.boundaries.exclude(*self.names),
            global_data=domain.global_data,
        )

    def extra_repr(self) -> str:
        return f"names={sorted(self.names)}"
