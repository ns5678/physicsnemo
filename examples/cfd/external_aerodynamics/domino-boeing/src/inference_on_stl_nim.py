# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import os
import re
import time
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.spatial import KDTree
import pyvista as pv
from torch import nn
from hydra import compose, initialize
import vtk
from vtk.util import numpy_support

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import (
    unnormalize,
    create_directory,
    nd_interpolator,
    get_filenames,
    write_to_vtp,
    write_to_vtu,
    get_volume_data,
)
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel
from physicsnemo.distributed import DistributedManager

import warp as wp

try:
    import cuml

    CUML_AVAILABLE = True
except:
    CUML_AVAILABLE = False

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except:
    CUPY_AVAILABLE = False
    cp = None

## Reference parameters to be accessed globally -- Not a good practice
## Ok for demo
PREF = np.float32(176.352)
UINFTY = np.float32(2679.505)


@wp.kernel
def _bvh_query_distance(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
    sdf_hit_point: wp.array(dtype=wp.vec3f),
    sdf_hit_point_id: wp.array(dtype=wp.int32),
):
    """
    Computes the signed distance from each point in the given array `points`
    to the mesh represented by `mesh`,within the maximum distance `max_dist`,
    and stores the result in the array `sdf`.

    Parameters:
        mesh (wp.uint64): The identifier of the mesh.
        points (wp.array): An array of 3D points for which to compute the
            signed distance.
        max_dist (wp.float32): The maximum distance within which to search
            for the closest point on the mesh.
        sdf (wp.array): An array to store the computed signed distances.
        sdf_hit_point (wp.array): An array to store the computed hit points.
        sdf_hit_point_id (wp.array): An array to store the computed hit point ids.

    Returns:
        None
    """
    tid = wp.tid()

    res = wp.mesh_query_point_sign_winding_number(mesh, points[tid], max_dist)

    mesh_ = wp.mesh_get(mesh)

    p0 = mesh_.points[mesh_.indices[3 * res.face + 0]]
    p1 = mesh_.points[mesh_.indices[3 * res.face + 1]]
    p2 = mesh_.points[mesh_.indices[3 * res.face + 2]]

    p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2

    sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))
    sdf_hit_point[tid] = p_closest
    sdf_hit_point_id[tid] = res.face


class SignedDistanceFieldModule(nn.Module):
    def __init__(
        self,
        mesh_vertices: Union[np.ndarray, torch.Tensor],
        mesh_indices: Union[np.ndarray, torch.Tensor],
        max_dist: float = 1e8,
        device: Optional[str] = None,
    ):
        """
        A torch.nn.Module that wraps Warp's signed_distance_field utility for inference.

        Args:
            mesh_vertices: (V, 3) float32 array or tensor of mesh vertex positions.
            mesh_indices: (F, 3) int32 array or tensor of triangle indices.
            max_dist: Maximum distance for SDF computation.
            device: Warp device string (e.g., "cuda:0") or None for default.
        """
        super().__init__()
        wp.init()

        self.device = device
        self.max_dist = max_dist

        # Convert mesh to Warp format
        mesh_vertices = (
            mesh_vertices.cpu().numpy()
            if isinstance(mesh_vertices, torch.Tensor)
            else mesh_vertices
        )
        mesh_indices = (
            mesh_indices.cpu().numpy()
            if isinstance(mesh_indices, torch.Tensor)
            else mesh_indices
        )

        self.mesh = wp.Mesh(
            wp.array(mesh_vertices, dtype=wp.vec3),
            wp.array(mesh_indices.flatten(), dtype=wp.int32),
        )

    def forward(
        self,
        input_points: Union[np.ndarray, torch.Tensor],
        include_hit_points: bool = False,
        include_hit_points_id: bool = False,
    ):
        """
        Args:
            input_points: (N, 3) float32 array or tensor of points.
            include_hit_points: Whether to return hit points.
            include_hit_points_id: Whether to return hit point IDs.
        Returns:
            torch.Tensor or tuple of tensors
        """
        if isinstance(input_points, torch.Tensor):
            input_points = input_points.cpu().numpy()

        sdf_points = wp.array(input_points, dtype=wp.vec3)
        sdf = wp.zeros(shape=sdf_points.shape, dtype=wp.float32)
        sdf_hit_point = wp.zeros(shape=sdf_points.shape, dtype=wp.vec3f)
        sdf_hit_point_id = wp.zeros(shape=sdf_points.shape, dtype=wp.int32)

        wp.launch(
            kernel=_bvh_query_distance,
            dim=len(sdf_points),
            inputs=[
                self.mesh.id,
                sdf_points,
                self.max_dist,
                sdf,
                sdf_hit_point,
                sdf_hit_point_id,
            ],
        )

        # Convert results to torch tensors
        sdf = wp.to_torch(sdf)

        if include_hit_points and include_hit_points_id:
            return sdf, wp.to_torch(sdf_hit_point), wp.to_torch(sdf_hit_point_id)
        elif include_hit_points:
            return sdf, wp.to_torch(sdf_hit_point)
        elif include_hit_points_id:
            return sdf, wp.to_torch(sdf_hit_point_id)
        else:
            return sdf


def shuffle_array_torch(surface_vertices, geometry_points, device):
    idx = torch.unsqueeze(
        torch.randperm(surface_vertices.shape[0])[:geometry_points], -1
    ).to(device)
    idx = idx.repeat(1, 3)
    surface_sampled = torch.gather(surface_vertices, 0, idx)
    return surface_sampled


class inferenceDataPipe:
    def __init__(
        self,
        device: int = 0,
        grid_resolution: Optional[list] = [256, 96, 64],
        normalize_coordinates: bool = False,
        geom_points_sample: int = 300000,
        positional_encoding: bool = False,
        surface_vertices=None,
        surface_indices=None,
        surface_areas=None,
        surface_centers=None,
        use_sdf_basis=False,
        gpu_preprocessing=True,
    ):
        self.surface_vertices = surface_vertices
        self.surface_indices = surface_indices
        self.surface_areas = surface_areas
        self.surface_centers = surface_centers
        self.device = device
        self.grid_resolution = grid_resolution
        self.normalize_coordinates = normalize_coordinates
        self.geom_points_sample = geom_points_sample
        self.positional_encoding = positional_encoding
        self.use_sdf_basis = use_sdf_basis
        self.array_provider = cp if gpu_preprocessing else np
        torch.manual_seed(int(42 + torch.cuda.current_device()))
        self.data_dict = {}

    def clear_dict(self):
        del self.data_dict

    def clear_volume_dict(self):
        del self.data_dict["volume_mesh_centers"]
        del self.data_dict["pos_enc_closest"]
        del self.data_dict["pos_normals_com"]
        del self.data_dict["sdf_nodes"]

    def _generate_volume_coordinates(
        self,
        num_pts_vol,
        c_min,
        c_max,
        bounding_box_nested=None,
        bounding_box_percentages=None,
        sample_nested=False,
    ):
        """
        Generate volume coordinates with optional hierarchical nested sampling.

        Args:
            num_pts_vol: Number of points to generate
            c_min: Minimum coordinates of main bounding box
            c_max: Maximum coordinates of main bounding box
            bounding_box_nested: Optional list of nested bounding boxes (tightest to loosest)
            bounding_box_percentages: Optional list of percentages for each nested box
            sample_nested: Whether to use hierarchical nested sampling

        Returns:
            torch.Tensor: Generated volume coordinates
        """
        # Hierarchical nested sampling with configurable percentages
        if (
            sample_nested
            and bounding_box_nested is not None
            and len(bounding_box_nested) > 0
        ):
            num_boxes = len(bounding_box_nested)

            # Use provided percentages or default to equal distribution
            if (
                bounding_box_percentages is not None
                and len(bounding_box_percentages) == num_boxes
            ):
                percentages = bounding_box_percentages
            else:
                num_regions = num_boxes + 1
                equal_pct = 100.0 / num_regions
                percentages = [equal_pct] * num_boxes

            # Calculate outermost region percentage
            total_pct = sum(percentages)
            outer_pct = 100.0 - total_pct

            all_samples = []
            # For each nested box level
            for i in range(num_boxes):
                box_min = bounding_box_nested[i][0]
                box_max = bounding_box_nested[i][1]
                pct = percentages[i]
                num_pts_level = int(num_pts_vol * pct / 100.0)

                if i == 0:
                    # Innermost box: sample uniformly inside
                    samples_level = (box_max - box_min) * torch.rand(
                        num_pts_level, 3, device=self.device, dtype=torch.float32
                    ) + box_min
                    all_samples.append(samples_level)
                else:
                    # Sample outside previous box but inside current box
                    prev_box_min = bounding_box_nested[i - 1][0]
                    prev_box_max = bounding_box_nested[i - 1][1]
                    samples_level = self._rejection_sample_between_boxes(
                        num_pts_level, box_min, box_max, prev_box_min, prev_box_max
                    )
                    all_samples.append(samples_level)

            # Sample outermost region: outside last nested box but inside main bbox
            num_pts_remaining = num_pts_vol - sum([s.shape[0] for s in all_samples])
            last_box_min = bounding_box_nested[-1][0]
            last_box_max = bounding_box_nested[-1][1]
            samples_outer = self._rejection_sample_between_boxes(
                num_pts_remaining, c_min, c_max, last_box_min, last_box_max
            )
            all_samples.append(samples_outer)

            # Combine all samples
            volume_coordinates_sub = torch.cat(all_samples, dim=0)

            # Shuffle to randomize order of points from different nested levels
            shuffle_idx = torch.randperm(
                volume_coordinates_sub.shape[0], device=self.device
            )
            volume_coordinates_sub = volume_coordinates_sub[shuffle_idx]
        else:
            # Original sampling: uniform in main bounding box
            volume_coordinates_sub = (c_max - c_min) * torch.rand(
                num_pts_vol, 3, device=self.device, dtype=torch.float32
            ) + c_min

        return volume_coordinates_sub

    def _rejection_sample_between_boxes(
        self,
        num_pts_target,
        outer_box_min,
        outer_box_max,
        inner_box_min,
        inner_box_max,
    ):
        """
        Sample points in outer box but excluding inner box using rejection sampling.

        Args:
            num_pts_target: Number of points to generate
            outer_box_min: Minimum coordinates of outer box
            outer_box_max: Maximum coordinates of outer box
            inner_box_min: Minimum coordinates of inner box (to exclude)
            inner_box_max: Maximum coordinates of inner box (to exclude)

        Returns:
            torch.Tensor: Sampled coordinates
        """
        # Handle edge case where no points are needed
        if num_pts_target <= 0:
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)
        
        samples_list = []
        num_collected = 0

        while num_collected < num_pts_target:
            # Oversample by 2x to account for rejections
            num_to_sample = int(2.0 * (num_pts_target - num_collected))
            
            candidates = (outer_box_max - outer_box_min) * torch.rand(
                num_to_sample, 3, device=self.device, dtype=torch.float32
            ) + outer_box_min

            # Filter out points inside inner bounding box
            outside_inner = (
                (candidates[:, 0] < inner_box_min[0])
                | (candidates[:, 0] > inner_box_max[0])
                | (candidates[:, 1] < inner_box_min[1])
                | (candidates[:, 1] > inner_box_max[1])
                | (candidates[:, 2] < inner_box_min[2])
                | (candidates[:, 2] > inner_box_max[2])
            )

            valid_candidates = candidates[outside_inner]
            num_valid = valid_candidates.shape[0]

            if num_valid > 0:
                num_to_add = min(num_valid, num_pts_target - num_collected)
                samples_list.append(valid_candidates[:num_to_add])
                num_collected += num_to_add
        
        # Handle edge case where no valid samples were collected
        if len(samples_list) == 0:
            return torch.zeros((0, 3), device=self.device, dtype=torch.float32)

        return torch.cat(samples_list, dim=0)

    def create_grid_torch(self, mx, mn, nres):
        start_time = time.time()
        dx = torch.linspace(mn[0], mx[0], nres[0], device=self.device)
        dy = torch.linspace(mn[1], mx[1], nres[1], device=self.device)
        dz = torch.linspace(mn[2], mx[2], nres[2], device=self.device)

        xv, yv, zv = torch.meshgrid(dx, dy, dz, indexing="ij")
        xv = torch.unsqueeze(xv, -1)
        yv = torch.unsqueeze(yv, -1)
        zv = torch.unsqueeze(zv, -1)
        grid = torch.cat((xv, yv, zv), axis=-1)
        return grid

    def process_surface_mesh(self, bounding_box=None, bounding_box_surface=None):
        # Use coarse mesh to calculate SDF
        surface_vertices = self.surface_vertices
        surface_indices = self.surface_indices
        surface_areas = self.surface_areas
        surface_centers = self.surface_centers

        start_time = time.time()

        if bounding_box is None:
            # Create a bounding box
            s_max = torch.amax(surface_vertices, 0)
            s_min = torch.amin(surface_vertices, 0)

            c_max = s_max + (s_max - s_min) / 2
            c_min = s_min - (s_max - s_min) / 2
            c_min[2] = s_min[2]
        else:
            c_min = bounding_box[0]
            c_max = bounding_box[1]

        if bounding_box_surface is None:
            # Create a bounding box
            s_max = torch.amax(surface_vertices, 0)
            s_min = torch.amin(surface_vertices, 0)

            surf_max = s_max + (s_max - s_min) / 2
            surf_min = s_min - (s_max - s_min) / 2
            surf_min[2] = s_min[2]
        else:
            surf_min = bounding_box_surface[0]
            surf_max = bounding_box_surface[1]

        nx, ny, nz = self.grid_resolution

        grid = self.create_grid_torch(c_max, c_min, self.grid_resolution)
        grid_reshaped = torch.reshape(grid, (nx * ny * nz, 3))

        # SDF on grid
        sdf_module = SignedDistanceFieldModule(
            surface_vertices, surface_indices, device=self.device
        ).to(self.device)
        sdf_grid = sdf_module(grid_reshaped)
        sdf_grid = torch.reshape(sdf_grid, (nx, ny, nz))

        surface_areas = torch.unsqueeze(surface_areas, -1)
        center_of_mass = torch.sum(surface_centers * surface_areas, 0) / torch.sum(
            surface_areas
        )

        s_grid = self.create_grid_torch(surf_max, surf_min, self.grid_resolution)
        surf_grid_reshaped = torch.reshape(s_grid, (nx * ny * nz, 3))

        surf_sdf_grid = sdf_module(surf_grid_reshaped)

        surf_sdf_grid = torch.reshape(surf_sdf_grid, (nx, ny, nz))

        if self.normalize_coordinates:
            grid = 2.0 * (grid - c_min) / (c_max - c_min) - 1.0
            s_grid = 2.0 * (s_grid - surf_min) / (surf_max - surf_min) - 1.0

        surface_vertices = torch.unsqueeze(surface_vertices, 0)
        grid = torch.unsqueeze(grid, 0)
        s_grid = torch.unsqueeze(s_grid, 0)
        sdf_grid = torch.unsqueeze(sdf_grid, 0)
        surf_sdf_grid = torch.unsqueeze(surf_sdf_grid, 0)
        max_min = [c_min, c_max]
        surf_max_min = [surf_min, surf_max]
        center_of_mass = center_of_mass

        return (
            surface_vertices,
            grid,
            sdf_grid,
            max_min,
            s_grid,
            surf_sdf_grid,
            surf_max_min,
            center_of_mass,
        )

    def sample_stl_points(
        self,
        num_points,
        stl_centers,
        stl_area,
        stl_normals,
        max_min,
        center_of_mass,
        bounding_box=None,
        stencil_size=7,
    ):
        if bounding_box is not None:
            c_max = bounding_box[1]
            c_min = bounding_box[0]
        else:
            c_min = max_min[0]
            c_max = max_min[1]

        start_time = time.time()

        nx, ny, nz = self.grid_resolution

        idx = np.arange(stl_centers.shape[0])

        if num_points is not None:
            idx = idx[:num_points]

        surface_coordinates = stl_centers
        surface_normals = stl_normals
        surface_area = stl_area

        if stencil_size > 1:
            if self.array_provider == cp and CUML_AVAILABLE:
                knn = cuml.neighbors.NearestNeighbors(
                    n_neighbors=stencil_size,
                    algorithm="rbc",
                )
                knn.fit(surface_coordinates)
                ii = knn.kneighbors(surface_coordinates, return_distance=False)
            else:
                interp_func = KDTree(surface_coordinates)
                _, ii = interp_func.query(surface_coordinates, k=stencil_size)

            surface_neighbors = surface_coordinates[ii]
            surface_neighbors = surface_neighbors[:, 1:] + 1e-6
            surface_neighbors_normals = surface_normals[ii]
            surface_neighbors_normals = surface_neighbors_normals[:, 1:]
            surface_neighbors_area = surface_area[ii]
            surface_neighbors_area = surface_neighbors_area[:, 1:]
        else:
            surface_neighbors = np.expand_dims(surface_coordinates, 1) + 1e-6
            surface_neighbors_normals = np.expand_dims(surface_normals, 1)
            surface_neighbors_area = np.expand_dims(surface_area, 1)

        surface_coordinates = torch.from_numpy(surface_coordinates).to(self.device)
        surface_normals = torch.from_numpy(surface_normals).to(self.device)
        surface_area = torch.from_numpy(surface_area).to(self.device)
        surface_neighbors = torch.from_numpy(surface_neighbors).to(self.device)
        surface_neighbors_normals = torch.from_numpy(surface_neighbors_normals).to(
            self.device
        )
        surface_neighbors_area = torch.from_numpy(surface_neighbors_area).to(
            self.device
        )

        pos_normals_com = surface_coordinates - center_of_mass

        if self.normalize_coordinates:
            surface_coordinates = (
                2.0 * (surface_coordinates - c_min) / (c_max - c_min) - 1.0
            )
            surface_neighbors = (
                2.0 * (surface_neighbors - c_min) / (c_max - c_min) - 1.0
            )

        surface_coordinates = surface_coordinates[idx]
        surface_area = surface_area[idx]
        surface_normals = surface_normals[idx]
        pos_normals_com = pos_normals_com[idx]
        surface_coordinates = torch.unsqueeze(surface_coordinates, 0)
        surface_normals = torch.unsqueeze(surface_normals, 0)
        surface_area = torch.unsqueeze(surface_area, 0)
        pos_normals_com = torch.unsqueeze(pos_normals_com, 0)

        surface_neighbors = surface_neighbors[idx]
        surface_neighbors_normals = surface_neighbors_normals[idx]
        surface_neighbors_area = surface_neighbors_area[idx]
        surface_neighbors = torch.unsqueeze(surface_neighbors, 0)
        surface_neighbors_normals = torch.unsqueeze(surface_neighbors_normals, 0)
        surface_neighbors_area = torch.unsqueeze(surface_neighbors_area, 0)

        scaling_factors = [c_max, c_min]

        return (
            surface_coordinates,
            surface_neighbors,
            surface_normals,
            surface_neighbors_normals,
            surface_area,
            surface_neighbors_area,
            pos_normals_com,
            scaling_factors,
            idx,
        )

    def sample_points_in_volume(
        self,
        num_points_vol=None,
        point_cloud=None,
        max_min=None,
        center_of_mass=None,
        bounding_box=None,
        bounding_box_nested=None,
        bounding_box_percentages=None,
        sample_nested=False,
    ):
        if bounding_box is not None:
            c_max = bounding_box[1]
            c_min = bounding_box[0]
        else:
            c_min = max_min[0]
            c_max = max_min[1]

        nx, ny, nz = self.grid_resolution

        if num_points_vol is not None:
            for k in range(10):
                print("k ", k)
                if k > 0:
                    num_pts_vol = num_points_vol - int(volume_coordinates.shape[0] / 2)
                else:
                    num_pts_vol = int(1.25 * num_points_vol)

                # Generate volume coordinates with optional hierarchical nested sampling
                volume_coordinates_sub = self._generate_volume_coordinates(
                    num_pts_vol,
                    c_min,
                    c_max,
                    bounding_box_nested,
                    bounding_box_percentages,
                    sample_nested,
                )

                sdf_module = SignedDistanceFieldModule(
                    self.surface_vertices, self.surface_indices, device=self.device
                ).to(self.device)
                sdf_nodes = sdf_module(
                    volume_coordinates_sub,
                    include_hit_points=False,
                )

                sdf_nodes = torch.unsqueeze(sdf_nodes, -1)

                idx = torch.unsqueeze(torch.where((sdf_nodes > 0))[0], -1)
                idx = idx.repeat(1, volume_coordinates_sub.shape[1])

                if k == 0:
                    volume_coordinates = torch.gather(volume_coordinates_sub, 0, idx)
                else:
                    volume_coordinates_1 = torch.gather(volume_coordinates_sub, 0, idx)
                    volume_coordinates = torch.cat(
                        (volume_coordinates, volume_coordinates_1), axis=0
                    )

                if volume_coordinates.shape[0] >= num_points_vol:
                    volume_coordinates = volume_coordinates[:num_points_vol]
                    break
        else:
            volume_coordinates = torch.from_numpy(np.float32(point_cloud)).to(
                self.device
            )

        sdf_module = SignedDistanceFieldModule(
            self.surface_vertices, self.surface_indices, device=self.device
        ).to(self.device)

        sdf_nodes, sdf_node_closest_point = sdf_module(
            volume_coordinates,
            include_hit_points=True,
        )
        sdf_nodes = torch.unsqueeze(sdf_nodes, -1)

        pos_normals_closest = volume_coordinates - sdf_node_closest_point
        pos_normals_com = volume_coordinates - center_of_mass

        if self.normalize_coordinates:
            volume_coordinates = (
                2.0 * (volume_coordinates - c_min) / (c_max - c_min) - 1.0
            )

        volume_coordinates = torch.unsqueeze(volume_coordinates, 0)
        pos_normals_com = torch.unsqueeze(pos_normals_com, 0)

        if self.use_sdf_basis:
            pos_normals_closest = torch.unsqueeze(pos_normals_closest, 0)
            sdf_nodes = torch.unsqueeze(sdf_nodes, 0)

        scaling_factors = [c_max, c_min]
        return (
            volume_coordinates,
            pos_normals_com,
            pos_normals_closest,
            sdf_nodes,
            scaling_factors,
        )


class dominoInference:
    def __init__(
        self,
        cfg: DictConfig,
        dist: None,
        cached_geo_encoding: False,
        device: Optional[str] = None,
    ):
        self.cfg = cfg
        self.dist = dist
        self.stream_velocity = None
        self.stencil_size = None
        self.stl_path = None
        self.stl_vertices = None
        self.stl_centers = None
        self.surface_areas = None
        self.mesh_indices_flattened = None
        self.length_scale = 1.0
        if self.dist is None:
            self.device = device
            self.world_size = 1
        else:
            self.device = self.dist.device
            self.world_size = self.dist.world_size

        # Initialize AoA (angle of attack) - this is the only variable parameter
        self.aoa = None
        self.aoa_reference = torch.full((1, 1), 22.0, dtype=torch.float32).to(
            self.device
        )

        self.num_vol_vars, self.num_surf_vars, self.num_global_features = (
            self.get_num_variables()
        )

        self.model = None
        self.grid_resolution = torch.tensor(self.cfg.model.interp_res).to(self.device)
        self.vol_factors = None
        self.bounding_box_min_max = None
        self.bounding_box_surface_min_max = None
        self.bounding_box_nested = None  # List of nested bounding boxes
        self.bounding_box_percentages = None  # List of percentages for each nested box
        self.center_of_mass = None
        self.grid = None
        self.geometry_encoding = None
        self.geometry_encoding_surface = None
        self.cached_geo_encoding = cached_geo_encoding
        self.out_dict = {}

    def get_geometry_encoding(self):
        return self.geometry_encoding

    def get_geometry_encoding_surface(self):
        return self.geometry_encoding_surface

    def get_out_dict(self):
        return self.out_dict

    def clear_out_dict(self):
        self.out_dict.clear()

    def initialize_data_processor(self):
        self.ifp = inferenceDataPipe(
            device=self.device,
            surface_vertices=self.stl_vertices,
            surface_indices=self.mesh_indices_flattened,
            surface_areas=self.surface_areas,
            surface_centers=self.stl_centers,
            grid_resolution=self.grid_resolution,
            normalize_coordinates=True,
            geom_points_sample=300000,
            positional_encoding=False,
            use_sdf_basis=self.cfg.model.use_sdf_in_basis_func,
        )

    def load_bounding_box(self):
        if (
            self.cfg.data.bounding_box.min is not None
            and self.cfg.data.bounding_box.max is not None
        ):
            c_min = torch.from_numpy(
                np.array(self.cfg.data.bounding_box.min, dtype=np.float32)
            ).to(self.device)
            c_max = torch.from_numpy(
                np.array(self.cfg.data.bounding_box.max, dtype=np.float32)
            ).to(self.device)
            self.bounding_box_min_max = [c_min, c_max]

        if (
            self.cfg.data.bounding_box_surface.min is not None
            and self.cfg.data.bounding_box_surface.max is not None
        ):
            c_min = torch.from_numpy(
                np.array(self.cfg.data.bounding_box_surface.min, dtype=np.float32)
            ).to(self.device)
            c_max = torch.from_numpy(
                np.array(self.cfg.data.bounding_box_surface.max, dtype=np.float32)
            ).to(self.device)
            self.bounding_box_surface_min_max = [c_min, c_max]

        # Load nested bounding boxes for hierarchical sampling
        if hasattr(self.cfg.data, "bounding_box_nested"):
            nested_boxes = []
            percentages = []

            # Dynamically iterate over all nested boxes
            nested_config = self.cfg.data.bounding_box_nested
            box_names = sorted(
                [name for name in dir(nested_config) if name.startswith("box")]
            )

            for box_name in box_names:
                box = getattr(nested_config, box_name)
                if hasattr(box, "min") and hasattr(box, "max"):
                    if box.min is not None and box.max is not None:
                        c_min = torch.from_numpy(
                            np.array(box.min, dtype=np.float32)
                        ).to(self.device)
                        c_max = torch.from_numpy(
                            np.array(box.max, dtype=np.float32)
                        ).to(self.device)
                        nested_boxes.append([c_min, c_max])

                        # Load percentage if available, otherwise use equal distribution
                        if hasattr(box, "percentage") and box.percentage is not None:
                            percentages.append(float(box.percentage))
                        else:
                            percentages.append(None)  # Will be computed later

            if len(nested_boxes) > 0:
                self.bounding_box_nested = nested_boxes

                # Handle percentages: if any are None, distribute equally
                if None in percentages:
                    num_regions = len(nested_boxes) + 1
                    equal_pct = 100.0 / num_regions
                    percentages = [equal_pct] * len(nested_boxes)

                self.bounding_box_percentages = percentages

                # Validate percentages sum to <= 100
                total_pct = sum(percentages)
                if total_pct > 100.0:
                    raise ValueError(
                        f"Sum of nested box percentages ({total_pct}%) exceeds 100%. "
                        f"Please adjust the percentages in config."
                    )

                print(
                    f"Loaded {len(nested_boxes)} nested bounding boxes with percentages: {percentages}"
                )
                print(
                    f"Outermost region will receive {100.0 - total_pct:.1f}% of points"
                )

    def load_volume_scaling_factors(self):
        # vol_mean = np.array(self.cfg.data.scaling_factors.volume.mean, dtype=np.float32)
        # vol_std = np.array(self.cfg.data.scaling_factors.volume.std, dtype=np.float32)
        # vol_factors = np.stack([vol_mean, vol_std])
        # vol_factors = torch.from_numpy(vol_factors).to(self.device)
        scaling_param_path = self.cfg.eval.scaling_param_path
        vol_factors_path = os.path.join(
            scaling_param_path, "volume_scaling_factors.npy"
        )

        vol_factors = np.load(vol_factors_path, allow_pickle=True)
        vol_factors = torch.from_numpy(vol_factors).to(self.device)

        return vol_factors

    def load_surface_scaling_factors(self):
        # surf_mean = np.array(
        #     self.cfg.data.scaling_factors.surface.mean, dtype=np.float32
        # )
        # surf_std = np.array(self.cfg.data.scaling_factors.surface.std, dtype=np.float32)
        # surf_factors = np.stack([surf_mean, surf_std])
        # surf_factors = torch.from_numpy(surf_factors).to(self.device)
        scaling_param_path = self.cfg.eval.scaling_param_path
        surf_factors_path = os.path.join(
            scaling_param_path, "surface_scaling_factors.npy"
        )
        surf_factors = np.load(surf_factors_path, allow_pickle=True)
        surf_factors = torch.from_numpy(surf_factors).to(self.device)

        return surf_factors

    def read_stl(self):
        reader = pv.get_reader(self.stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        stl_centers = mesh_stl.cell_centers().points
        # Assuming triangular elements
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[:, 1:]
        mesh_indices_flattened = stl_faces.flatten()

        surface_areas = mesh_stl.compute_cell_sizes(
            length=False, area=True, volume=False
        )
        surface_areas = np.array(surface_areas.cell_data["Area"])
        idx = np.where(surface_areas > 0.0)
        surface_normals = np.array(mesh_stl.cell_normals, dtype=np.float32)

        surface_areas = surface_areas[idx]
        stl_centers = stl_centers[idx]
        surface_normals = surface_normals[idx]

        self.stl_vertices = torch.from_numpy(np.float32(stl_vertices)).to(self.device)
        self.stl_centers = torch.from_numpy(np.float32(stl_centers)).to(self.device)
        self.surface_areas = torch.from_numpy(np.float32(surface_areas)).to(self.device)
        self.stl_normals = torch.from_numpy(np.float32(surface_normals)).to(self.device)
        self.mesh_indices_flattened = torch.from_numpy(
            np.int32(mesh_indices_flattened)
        ).to(self.device)
        self.length_scale = length_scale
        self.mesh_stl = mesh_stl

    def read_stl_trimesh(
        self, stl_vertices, stl_faces, stl_centers, surface_normals, surface_areas
    ):
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))

        idx = np.where(surface_areas > 0.0)
        surface_areas = surface_areas[idx]
        stl_centers = stl_centers[idx]
        surface_normals = surface_normals[idx]

        self.stl_vertices = torch.from_numpy(stl_vertices).to(self.device)
        self.stl_centers = torch.from_numpy(stl_centers).to(self.device)
        self.stl_normals = -1.0 * torch.from_numpy(surface_normals).to(self.device)
        self.surface_areas = torch.from_numpy(surface_areas).to(self.device)
        self.mesh_indices_flattened = torch.from_numpy(
            np.int32(mesh_indices_flattened)
        ).to(self.device)
        self.length_scale = length_scale

    def get_num_variables(self):
        volume_variable_names = list(self.cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in volume_variable_names:
            if self.cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1

        surface_variable_names = list(self.cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if self.cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1

        num_global_features = 0
        global_params_names = list(self.cfg.variables.global_parameters.keys())
        for param in global_params_names:
            if self.cfg.variables.global_parameters[param].type == "vector":
                num_global_features += len(
                    self.cfg.variables.global_parameters[param].reference
                )
            elif self.cfg.variables.global_parameters[param].type == "scalar":
                num_global_features += 1
            else:
                raise ValueError(f"Unknown global parameter type")

        return num_vol_vars, num_surf_vars, num_global_features

    def initialize_model(self, model_path):
        model = (
            DoMINO(
                input_features=3,
                output_features_vol=self.num_vol_vars,
                output_features_surf=self.num_surf_vars,
                global_features=self.num_global_features,
                model_parameters=self.cfg.model,
            )
            .to(self.device)
            .eval()
        )

        checkpoint_iter = torch.load(
            to_absolute_path(model_path), map_location=self.device
        )

        model.load_state_dict(checkpoint_iter)

        if self.dist is not None and self.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[self.dist.local_rank],
                output_device=self.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )

        self.model = model
        self.vol_factors = self.load_volume_scaling_factors()
        self.surf_factors = self.load_surface_scaling_factors()
        self.load_bounding_box()

    def set_stream_velocity(self, stream_velocity):
        self.stream_velocity = torch.full(
            (1, 1), stream_velocity, dtype=torch.float32
        ).to(self.device)

    def set_aoa(self, aoa):
        self.aoa = torch.full((1, 1), aoa, dtype=torch.float32).to(self.device)

    def set_stencil_size(self, stencil_size):
        self.stencil_size = stencil_size

    def set_stl_path(self, filename):
        self.stl_path = filename

    @torch.no_grad()
    def compute_geo_encoding(self, cached_geom_path=None):
        start_time = time.time()

        if not self.cached_geo_encoding:
            (
                surface_vertices,
                grid,
                sdf_grid,
                max_min,
                s_grid,
                surf_sdf_grid,
                surf_max_min,
                center_of_mass,
            ) = self.ifp.process_surface_mesh(
                self.bounding_box_min_max, self.bounding_box_surface_min_max
            )
            if self.bounding_box_min_max is None:
                self.bounding_box_min_max = max_min
            if self.bounding_box_surface_min_max is None:
                self.bounding_box_surface_min_max = surf_max_min
            self.center_of_mass = center_of_mass
            self.grid = grid
            self.s_grid = s_grid
            self.sdf_grid = sdf_grid
            self.surf_sdf_grid = surf_sdf_grid
            self.out_dict["sdf"] = sdf_grid

            geo_encoding, geo_encoding_surface = self.calculate_geometry_encoding(
                surface_vertices, grid, sdf_grid, s_grid, surf_sdf_grid, self.model
            )
        else:
            out_dict_cached = torch.load(cached_geom_path, map_location=self.device)
            self.bounding_box_min_max = out_dict_cached["bounding_box_min_max"]
            self.grid = out_dict_cached["grid"]
            self.sdf_grid = out_dict_cached["sdf_grid"]
            self.center_of_mass = out_dict_cached["com"]
            geo_encoding = out_dict_cached["geo_encoding"]
            geo_encoding_surface = out_dict_cached["geo_encoding_surface"]
            self.out_dict["sdf"] = self.sdf_grid
        print("Time taken for geo encoding = %f" % (time.time() - start_time))

        self.geometry_encoding = geo_encoding
        self.geometry_encoding_surface = geo_encoding_surface

        self.out_dict["bounding_box_dims"] = torch.vstack(self.bounding_box_min_max)

    def compute_forces(self):
        pressure = self.out_dict["pressure_surface"]
        wall_shear = self.out_dict["wall-shear-stress"]

        if self.surface_mesh is None:
            surface_normals = self.stl_normals[self.sampling_indices]
            surface_areas = self.surface_areas[self.sampling_indices]
        else:
            surface_areas = torch.tensor(
                self.surface_mesh["surface_mesh_areas"][self.sampling_indices]
            ).to(self.device)
            surface_normals = torch.tensor(
                self.surface_mesh["surface_mesh_normals"][self.sampling_indices]
            ).to(self.device)

        drag_force = torch.sum(
            pressure[0, :, 0] * surface_normals[:, 0] * surface_areas
            - wall_shear[0, :, 0] * surface_areas
        )
        lift_force = torch.sum(
            pressure[0, :, 0] * surface_normals[:, 2] * surface_areas
            - wall_shear[0, :, 2] * surface_areas
        )

        self.out_dict["drag_force"] = drag_force
        self.out_dict["lift_force"] = lift_force

    @torch.inference_mode()
    def compute_surface_solutions(
        self,
        num_sample_points=None,
        surface_mesh=None,
        plot_solutions=False,
        eval_batch_size=1_024_000,
    ):
        total_time = 0.0

        geo_encoding = self.geometry_encoding_surface

        if surface_mesh is not None:
            stl_coordinates = surface_mesh["surface_mesh_center_coordinates"]
            stl_areas = surface_mesh["surface_mesh_areas"]
            stl_normals = surface_mesh["surface_mesh_normals"]
            self.surface_mesh = surface_mesh
        else:
            stl_coordinates = self.stl_centers.cpu().numpy()
            stl_areas = self.surface_areas.cpu().numpy()
            stl_normals = self.stl_normals.cpu().numpy()
            self.surface_mesh = None

        with autocast(enabled=True):
            start_time = time.time()
            (
                surface_mesh_centers,
                surface_neighbors,
                surface_normals,
                surface_neighbors_normals,
                surface_areas,
                surface_neighbors_areas,
                pos_normals_com,
                surf_scaling_factors,
                sampling_indices,
            ) = self.ifp.sample_stl_points(
                num_sample_points,
                stl_coordinates,
                stl_areas,
                stl_normals,
                max_min=self.bounding_box_surface_min_max,
                center_of_mass=self.center_of_mass,
                stencil_size=self.stencil_size,
            )
            cur_time = time.time() - start_time
            print(f"sample_points_in_surface time (s): {cur_time:.4f}")

            surface_coordinates_all = surface_mesh_centers

            inner_time = time.time()
            start_time = time.time()

            if num_sample_points is None:
                point_batch_size = eval_batch_size
                num_points = surface_coordinates_all.shape[1]
                subdomain_points = int(np.floor(num_points / point_batch_size))
                surface_solutions = torch.zeros(1, num_points, self.num_surf_vars).to(
                    self.device
                )
                for p in range(subdomain_points + 1):
                    start_idx = p * point_batch_size
                    end_idx = (p + 1) * point_batch_size
                    surface_solutions_batch = self.compute_solution_on_surface(
                        geo_encoding,
                        surface_mesh_centers[:, start_idx:end_idx],
                        surface_neighbors[:, start_idx:end_idx],
                        surface_normals[:, start_idx:end_idx],
                        surface_neighbors_normals[:, start_idx:end_idx],
                        surface_areas[:, start_idx:end_idx] + 1e-9,
                        surface_neighbors_areas[:, start_idx:end_idx],
                        pos_normals_com[:, start_idx:end_idx],
                        self.s_grid,
                        self.model,
                        aoa=self.aoa,
                    )
                    surface_solutions[:, start_idx:end_idx] = surface_solutions_batch
            else:
                point_batch_size = eval_batch_size
                num_points = num_sample_points
                subdomain_points = int(np.floor(num_points / point_batch_size))
                surface_solutions = torch.zeros(1, num_points, self.num_surf_vars).to(
                    self.device
                )
                for p in range(subdomain_points + 1):
                    start_idx = p * point_batch_size
                    end_idx = (p + 1) * point_batch_size
                    surface_solutions_batch = self.compute_solution_on_surface(
                        geo_encoding,
                        surface_mesh_centers[:, start_idx:end_idx],
                        surface_neighbors[:, start_idx:end_idx],
                        surface_normals[:, start_idx:end_idx],
                        surface_neighbors_normals[:, start_idx:end_idx],
                        surface_areas[:, start_idx:end_idx],
                        surface_neighbors_areas[:, start_idx:end_idx],
                        pos_normals_com[:, start_idx:end_idx],
                        self.s_grid,
                        self.model,
                        aoa=self.aoa,
                    )
                    surface_solutions[:, start_idx:end_idx] = surface_solutions_batch

            cur_time = time.time() - start_time
            print(f"Compute_solution time (s): {cur_time:.4f}")
            total_time += float(time.time() - inner_time)
            surface_solutions_all = surface_solutions
            print(
                "Time taken for compute solution on surface for=%f, %f"
                % (time.time() - inner_time, torch.cuda.utilization(self.device))
            )
        cmax = surf_scaling_factors[0]
        cmin = surf_scaling_factors[1]

        surface_coordinates_all = torch.reshape(
            surface_coordinates_all, (1, num_points, 3)
        )
        surface_solutions_all = torch.reshape(
            surface_solutions_all, (1, num_points, self.num_surf_vars)
        )

        if self.surf_factors is not None:
            surface_solutions_all = unnormalize(
                surface_solutions_all, self.surf_factors[0], self.surf_factors[1]
            )

        self.out_dict["surface_coordinates"] = (
            0.5 * (surface_coordinates_all + 1.0) * (cmax - cmin) + cmin
        )
        self.out_dict["pressure_surface"] = surface_solutions_all[:, :, :1] * PREF
        self.out_dict["wall-shear-stress"] = surface_solutions_all[:, :, 1:] * PREF
        self.sampling_indices = sampling_indices

    @torch.inference_mode()
    def compute_volume_solutions(
        self,
        num_sample_points=None,
        point_cloud=None,
        point_cloud_sampled=None,
        plot_solutions=False,
        eval_batch_size=1_024_000,
        sample_nested=False,
    ):
        if (num_sample_points is None and point_cloud is None) or (
            num_sample_points is not None and point_cloud is not None
        ):
            raise ValueError(
                "Please provide either number of sampling points or a point cloud"
            )

        total_time = 0.0

        geo_encoding = self.geometry_encoding

        point_batch_size = eval_batch_size

        if num_sample_points is not None:
            num_points = num_sample_points
        else:
            if point_cloud_sampled is not None:
                num_points = point_cloud_sampled
                idx = np.random.choice(
                    point_cloud.shape[0], point_cloud_sampled, replace=False
                )
                point_cloud = point_cloud[idx]
            else:
                num_points = point_cloud.shape[0]

        subdomain_points = int(np.floor(num_points / point_batch_size))
        volume_solutions = torch.zeros(1, num_points, self.num_vol_vars).to(self.device)
        volume_coordinates = torch.zeros(1, num_points, 3).to(self.device)

        for p in range(subdomain_points + 1):
            start_idx = p * point_batch_size
            end_idx = (p + 1) * point_batch_size
            if end_idx > num_points:
                point_batch_size = num_points - start_idx
                end_idx = num_points
            
            # Skip if no points to process in this batch
            if point_batch_size <= 0:
                break

            if point_cloud is not None:
                point_cloud_sub = point_cloud[start_idx:end_idx]
            # Compute volume
            with autocast(enabled=True):
                inner_time = time.time()
                start_time = time.time()
                if num_sample_points is not None:
                    (
                        volume_mesh_centers,
                        pos_normals_com,
                        pos_normals_closest,
                        sdf_nodes,
                        scaling_factors,
                    ) = self.ifp.sample_points_in_volume(
                        num_points_vol=point_batch_size,
                        max_min=self.bounding_box_min_max,
                        center_of_mass=self.center_of_mass,
                        bounding_box_nested=self.bounding_box_nested,
                        bounding_box_percentages=self.bounding_box_percentages,
                        sample_nested=sample_nested,
                    )
                else:
                    (
                        volume_mesh_centers,
                        pos_normals_com,
                        pos_normals_closest,
                        sdf_nodes,
                        scaling_factors,
                    ) = self.ifp.sample_points_in_volume(
                        num_points_vol=None,
                        point_cloud=point_cloud_sub,
                        max_min=self.bounding_box_min_max,
                        center_of_mass=self.center_of_mass,
                    )
                cur_time = time.time() - start_time
                print(f"sample_points_in_volume time (s): {cur_time:.4f}")

                volume_coordinates[:, start_idx:end_idx] = volume_mesh_centers

                # start_event.record()
                start_time = time.time()
                volume_solutions_batch = self.compute_solution_in_volume(
                    geo_encoding,
                    volume_mesh_centers,
                    sdf_nodes,
                    pos_normals_closest,
                    pos_normals_com,
                    self.grid,
                    self.model,
                    use_sdf_basis=self.cfg.model.use_sdf_in_basis_func,
                    aoa=self.aoa,
                )
                volume_solutions[:, start_idx:end_idx] = volume_solutions_batch

                cur_time = time.time() - start_time
                print(f"Compute_solution time (s): {cur_time:.4f}")
                total_time += float(time.time() - inner_time)
                print(
                    "Time taken for compute solution in volume for =%f, %f"
                    % (time.time() - inner_time, torch.cuda.utilization(self.device))
                )
        print("Total time measured = %f" % total_time)

        cmax = scaling_factors[0]
        cmin = scaling_factors[1]

        volume_coordinates_all = volume_coordinates
        volume_solutions_all = volume_solutions

        volume_coordinates_all = torch.reshape(
            volume_coordinates_all, (1, num_points, 3)
        )
        volume_solutions_all = torch.reshape(
            volume_solutions_all, (1, num_points, self.num_vol_vars)
        )

        if self.vol_factors is not None:
            volume_solutions_all = unnormalize(
                volume_solutions_all, self.vol_factors[0], self.vol_factors[1]
            )
        self.out_dict["coordinates"] = (
            0.5 * (volume_coordinates_all + 1.0) * (cmax - cmin) + cmin
        )

        self.out_dict["pressure"] = volume_solutions_all[:, :, 0:1] * PREF
        self.out_dict["velocity"] = volume_solutions_all[:, :, 1:4] * UINFTY
        self.out_dict["turbulent-kinetic-energy"] = self.out_dict["pressure"]
        self.out_dict["turbulent-viscosity"] = self.out_dict["pressure"]

    def cold_start(self, cached_geom_path=None):
        print("Cold start")
        self.compute_geo_encoding(cached_geom_path)
        self.compute_volume_solutions(num_sample_points=10)
        self.clear_out_dict()

    @torch.no_grad()
    def calculate_geometry_encoding(
        self, geo_centers, p_grid, sdf_grid, s_grid, sdf_surf_grid, model
    ):
        vol_min = self.bounding_box_min_max[0]
        vol_max = self.bounding_box_min_max[1]
        surf_min = self.bounding_box_surface_min_max[0]
        surf_max = self.bounding_box_surface_min_max[1]

        geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
        if self.world_size == 1:
            encoding_g_vol = model.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)
        else:
            encoding_g_vol = model.module.geo_rep_volume(
                geo_centers_vol, p_grid, sdf_grid
            )

        geo_centers_surf = 2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1

        if self.world_size == 1:
            encoding_g_surf = model.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )
        else:
            encoding_g_surf = model.module.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        return 0.5 * encoding_g_vol, 0.5 * encoding_g_surf

    @torch.no_grad()
    def compute_solution_on_surface(
        self,
        geo_encoding,
        surface_mesh_centers,
        surface_mesh_neighbors,
        surface_normals,
        surface_neighbors_normals,
        surface_areas,
        surface_neighbors_areas,
        pos_normals_com,
        s_grid,
        model,
        aoa,
    ):
        global_params_values = aoa
        global_params_values = torch.unsqueeze(global_params_values, -1)  # (1, 1, 1)

        global_params_reference = self.aoa_reference
        global_params_reference = torch.unsqueeze(
            global_params_reference, -1
        )  # (1, 1, 1)

        if self.world_size == 1:
            geo_encoding_local = model.geo_encoding_local(
                geo_encoding, surface_mesh_centers, s_grid, mode="surface"
            )
        else:
            geo_encoding_local = model.module.geo_encoding_local(
                geo_encoding, surface_mesh_centers, s_grid, mode="surface"
            )
        pos_encoding = pos_normals_com
        surface_areas = torch.unsqueeze(surface_areas, -1)
        surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)

        if self.world_size == 1:
            pos_encoding = model.position_encoder(pos_encoding, eval_mode="surface")
            tpredictions_batch = model.calculate_solution_with_neighbors(
                surface_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                surface_mesh_neighbors,
                surface_normals,
                surface_neighbors_normals,
                surface_areas,
                surface_neighbors_areas,
                global_params_values,
                global_params_reference,
                num_sample_points=self.stencil_size,
            )
        else:
            pos_encoding = model.module.position_encoder(
                pos_encoding, eval_mode="surface"
            )
            tpredictions_batch = model.module.calculate_solution_with_neighbors(
                surface_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                surface_mesh_neighbors,
                surface_normals,
                surface_neighbors_normals,
                surface_areas,
                surface_neighbors_areas,
                global_params_values,
                global_params_reference,
                num_sample_points=self.stencil_size,
            )

        return tpredictions_batch

    @torch.no_grad()
    def compute_solution_in_volume(
        self,
        geo_encoding,
        volume_mesh_centers,
        sdf_nodes,
        pos_enc_closest,
        pos_normals_com,
        p_grid,
        model,
        use_sdf_basis,
        aoa,
    ):
        ## Global parameters
        global_params_values = aoa
        global_params_values = torch.unsqueeze(global_params_values, -1)  # (1, 1, 1)

        global_params_reference = self.aoa_reference  # (1, 1)
        global_params_reference = torch.unsqueeze(
            global_params_reference, -1
        )  # (1, 1, 1)

        if self.world_size == 1:
            geo_encoding_local = model.geo_encoding_local(
                geo_encoding, volume_mesh_centers, p_grid, mode="volume"
            )
        else:
            geo_encoding_local = model.module.geo_encoding_local(
                geo_encoding, volume_mesh_centers, p_grid, mode="volume"
            )
        if use_sdf_basis:
            pos_encoding = torch.cat(
                (sdf_nodes, pos_enc_closest, pos_normals_com), axis=-1
            )
        else:
            pos_encoding = pos_normals_com

        if self.world_size == 1:
            pos_encoding = model.position_encoder(pos_encoding, eval_mode="volume")
            tpredictions_batch = model.calculate_solution(
                volume_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                global_params_values,
                global_params_reference,
                num_sample_points=self.stencil_size,
                eval_mode="volume",
            )
        else:
            pos_encoding = model.module.position_encoder(
                pos_encoding, eval_mode="volume"
            )
            tpredictions_batch = model.module.calculate_solution(
                volume_mesh_centers,
                geo_encoding_local,
                pos_encoding,
                global_params_values,
                global_params_reference,
                num_sample_points=self.stencil_size,
                eval_mode="volume",
            )
        return tpredictions_batch


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    with initialize(version_base="1.3", config_path="conf"):
        cfg = compose(config_name="config")

    DistributedManager.initialize()
    dist = DistributedManager()

    if dist.world_size > 1:
        torch.distributed.barrier()

    input_path = cfg.eval.test_path
    dirnames = get_filenames(input_path)
    dev_id = torch.cuda.current_device()
    num_files = int(len(dirnames) / 1)
    dirnames_per_gpu = dirnames[int(num_files * dev_id) : int(num_files * (dev_id + 1))]

    # Output directory for predictions (following test.py pattern)
    pred_save_path = cfg.eval.save_path

    domino = dominoInference(cfg, dist, False)

    # Load model checkpoint from config (following test.py pattern)
    model_path = os.path.join(cfg.resume_dir, cfg.eval.checkpoint_name)
    domino.initialize_model(model_path=model_path)

    for count, dirname in enumerate(dirnames_per_gpu):
        print(f"Processing sample {dirname}")
        filepath = os.path.join(input_path, dirname)

        # Extract tag for output naming (e.g., "LHC001_AoA_22")
        tag = re.findall(r"(LHC\d{3}_AoA_\d+)", dirname)[0]

        # Input STL file path following test.py pattern
        stl_filepath = os.path.join(filepath, f"{dirname}.stl")
        vtu_filepath = os.path.join(filepath, "volume_" + f"{dirname}.vtu")

        # Extract AoA from directory name
        aoa_match = re.search(r"AoA_(\d+(?:\.\d+)?)", dirname)
        if aoa_match:
            AOA = np.float32(aoa_match.group(1))
        else:
            raise ValueError(f"Could not extract AoA from folder name: {dirname}")

        STENCIL_SIZE = 20  # 20 is default value in test.py
        STREAM_VELOCITY = AOA  # Using AoA value as stream velocity proxy

        domino.set_stl_path(stl_filepath)
        ## TODO: Remove one of these later
        domino.set_stream_velocity(STREAM_VELOCITY)
        domino.set_aoa(AOA)  # Set AoA explicitly for global parameters

        domino.set_stencil_size(STENCIL_SIZE)

        #### Get the unstructured grid data for VTU output
        # reader = vtk.vtkXMLUnstructuredGridReader()
        # reader.SetFileName(vtu_filepath)
        # reader.Update()
        # polydata = reader.GetOutput()
        # volume_coordinates, volume_fields = get_volume_data(
        #     polydata, cfg.variables.volume.solution.keys()
        # )
        # volume_fields = np.concatenate(volume_fields, axis=-1)
        # c_min = cfg.data.bounding_box.min
        # c_max = cfg.data.bounding_box.max
        # ids_in_bbox = np.where(
        #     (volume_coordinates[:, 0] < c_min[0])
        #     | (volume_coordinates[:, 0] > c_max[0])
        #     | (volume_coordinates[:, 1] < c_min[1])
        #     | (volume_coordinates[:, 1] > c_max[1])
        #     | (volume_coordinates[:, 2] < c_min[2])
        #     | (volume_coordinates[:, 2] > c_max[2])
        # )

        domino.read_stl()
        domino.initialize_data_processor()
        domino.compute_geo_encoding()
        domino.compute_surface_solutions()

        ### Calculate volume solutions

        ## For NIM deployment with hierarchical nested sampling
        domino.compute_volume_solutions(
            num_sample_points=1_024_000, plot_solutions=False, sample_nested=True
        )

        ## For validation with predefined test VTU file
        # domino.compute_volume_solutions(
        #     num_sample_points=None, point_cloud=volume_coordinates,
        #     point_cloud_sampled=90_000_000, plot_solutions=False
        # )

        # domino.compute_forces()

        out_dict = domino.get_out_dict()

        surface_variable_names = list(cfg.variables.surface.solution.keys())
        volume_variable_names = list(cfg.variables.volume.solution.keys())

        vtp_out_path = os.path.join(pred_save_path, f"boundary_{tag}_predicted.vtp")
        npz_out_path = os.path.join(
            pred_save_path, f"volume_{tag}_predicted_1M_nested_hierarchical_sampled.npz"
        )

        # ===== WRITE SURFACE VTU (following test.py pattern) =====
        # Use the mesh_stl from domino (pyvista mesh), add predictions as cell data
        mesh_surf = domino.mesh_stl.copy()

        # Add prediction arrays to mesh cell data (following test.py pattern)
        mesh_surf[f"{surface_variable_names[0]}Pred"] = (
            out_dict["pressure_surface"][0].cpu().numpy().astype(np.float32)
        )
        mesh_surf[f"{surface_variable_names[1]}Pred"] = (
            out_dict["wall-shear-stress"][0].cpu().numpy().astype(np.float32)
        )

        # Convert back to point data before saving (following test.py pattern)
        mesh_surf_with_point_data = mesh_surf.cell_data_to_point_data()

        # Save the mesh with predictions as VTU (using point data)
        mesh_surf_with_point_data.save(vtp_out_path)
        print(f"Write surface VTU done for {tag}")

        # ===== WRITE VOLUME VTU (following test.py pattern) =====
        # Create a clean pyvista PointCloud with volume coordinates and predictions only (no ground truth)
        volume_coords = out_dict["coordinates"][0].cpu().numpy().astype(np.float32)
        volume_pressure = out_dict["pressure"][0].cpu().numpy().astype(np.float32)
        volume_velocity = out_dict["velocity"][0].cpu().numpy().astype(np.float32)

        # Zero out predictions outside bounding box (optional filter)
        c_min = cfg.data.bounding_box.min
        c_max = cfg.data.bounding_box.max
        ids_in_bbox = np.where(
            (volume_coords[:, 0] < c_min[0])
            | (volume_coords[:, 0] > c_max[0])
            | (volume_coords[:, 1] < c_min[1])
            | (volume_coords[:, 1] > c_max[1])
            | (volume_coords[:, 2] < c_min[2])
            | (volume_coords[:, 2] > c_max[2])
        )
        volume_pressure[ids_in_bbox] = 0.0
        volume_velocity[ids_in_bbox] = 0.0

        # Save the volume npz with predictions
        vol_dict = {}
        vol_dict["coordinates"] = volume_coords
        vol_dict["pressure"] = volume_pressure
        vol_dict["velocity"] = volume_velocity
        np.savez(npz_out_path, **vol_dict)

        print(f"Write volume NPZ done for {tag}")

    exit()
