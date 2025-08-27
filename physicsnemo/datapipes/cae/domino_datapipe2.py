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

"""
This code provides the datapipe for reading the processed npy files,
generating multi-res grids, calculating signed distance fields,
positional encodings, sampling random points in the volume and on surface,
normalizing fields and returning the output tensors as a dictionary.

This datapipe also non-dimensionalizes the fields, so the order in which the variables should
be fixed: velocity, pressure, turbulent viscosity for volume variables and
pressure, wall-shear-stress for surface variables. The different parameters such as
variable names, domain resolution, sampling size etc. are configurable in config.yaml.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Protocol, Sequence, Union

import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset

from physicsnemo.datapipes.cae.drivaer_ml_dataset import (
    DrivaerMLDataset,
    compute_mean_std_min_max,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.domino.utils import (
    ArrayType,
    area_weighted_shuffle_array,
    calculate_center_of_mass,
    calculate_normal_positional_encoding,
    create_grid,
    get_filenames,
    normalize,
    pad,
    shuffle_array,
    standardize,
)
from physicsnemo.utils.neighbors import knn
from physicsnemo.utils.profiling import profile
from physicsnemo.utils.sdf import signed_distance_field


class BoundingBox(Protocol):
    """
    Type definition for the required format of bounding box dimensions.
    """

    min: ArrayType
    max: ArrayType


@dataclass
class DoMINODataConfig:
    """Configuration for DoMINO dataset processing pipeline.

    Attributes:
        data_path: Path to the dataset to load.
        phase: Which phase of data to load ("train", "val", or "test").
        surface_variables: (Surface specific) Names of surface variables.
        surface_points_sample: (Surface specific) Number of surface points to sample per batch.
        num_surface_neighbors: (Surface specific) Number of surface neighbors to consider for nearest neighbors approach.
        resample_surfaces: (Surface specific) Whether to resample the surface before kdtree/knn. Not available if caching.
        resampling_points: (Surface specific) Number of points to resample the surface to.
        surface_sampling_algorithm: (Surface specific) Algorithm to use for surface sampling ("area_weighted" or "random").
        surface_factors: (Surface specific) Non-dimensionalization factors for surface variables.
            If set, and scaling_type is:
            - min_max_scaling -> rescale surface_fields to the min/max set here
            - mean_std_scaling -> rescale surface_fields to the mean and std set here.
        bounding_box_dims_surf: (Surface specific) Dimensions of bounding box. Must be an object with min/max
            attributes that are arraylike.
        volume_variables: (Volume specific) Names of volume variables.
        volume_points_sample: (Volume specific) Number of volume points to sample per batch.
        volume_factors: (Volume specific) Non-dimensionalization factors for volume variables scaling.
            If set, and scaling_type is:
            - min_max_scaling -> rescale volume_fields to the min/max set here
            - mean_std_scaling -> rescale volume_fields to the mean and std set here.
        bounding_box_dims: (Volume specific) Dimensions of bounding box. Must be an object with min/max
            attributes that are arraylike.
        grid_resolution: Resolution of the latent grid.
        normalize_coordinates: Whether to normalize coordinates based on min/max values.
            For surfaces: uses s_min/s_max, defined from:
            - Surface bounding box, if defined.
            - Min/max of the stl_vertices
            For volumes: uses c_min/c_max, defined from:
            - Volume bounding_box if defined,
            - 1.5x s_min/max otherwise, except c_min[2] = s_min[2] in this case
        sample_in_bbox: Whether to sample points in a specified bounding box.
            Uses the same min/max points as coordinate normalization.
            Only performed if compute_scaling_factors is false.
        sampling: Whether to downsample the full resolution mesh to fit in GPU memory.
            Surface and volume sampling points are configured separately as:
            - surface.points_sample
            - volume.points_sample
        geom_points_sample: Number of STL points sampled per batch.
            Independent of volume.points_sample and surface.points_sample.
        positional_encoding: Whether to use positional encoding. Affects the calculation of:
            - pos_volume_closest
            - pos_volume_center_of_mass
            - pos_surface_centter_of_mass
        scaling_type: Scaling type for volume variables.
            If used, will rescale the volume_fields and surface fields outputs.
            Requires volume.factor and surface.factor to be set.
        compute_scaling_factors: Whether to compute scaling factors.
            Not available if caching.
            Many preprocessing pieces are disabled if computing scaling factors.
        caching: Whether this is for caching or serving.
        deterministic: Whether to use a deterministic seed for sampling and random numbers.
        gpu_preprocessing: Whether to do preprocessing on the GPU (False for CPU).
        gpu_output: Whether to return output on the GPU as cupy arrays.
            If False, returns numpy arrays.
            You might choose gpu_preprocessing=True and gpu_output=False if caching.
    """

    data_path: Path
    phase: Literal["train", "val", "test"]

    # Surface-specific variables:
    surface_variables: Optional[Sequence] = ("pMean", "wallShearStress")
    surface_points_sample: int = 1024
    num_surface_neighbors: int = 11
    resample_surfaces: bool = False
    resampling_points: int = 1_000_000
    surface_sampling_algorithm: str = Literal["area_weighted", "random"]
    surface_factors: Optional[Sequence] = None
    bounding_box_dims_surf: Optional[Union[BoundingBox, Sequence]] = None

    # Volume specific variables:
    volume_variables: Optional[Sequence] = ("UMean", "pMean")
    volume_points_sample: int = 1024
    volume_factors: Optional[Sequence] = None
    bounding_box_dims: Optional[Union[BoundingBox, Sequence]] = None

    grid_resolution: Union[Sequence, ArrayType] = (256, 96, 64)
    normalize_coordinates: bool = False
    sample_in_bbox: bool = False
    sampling: bool = False
    geom_points_sample: int = 300000
    positional_encoding: bool = False
    scaling_type: Optional[Literal["min_max_scaling", "mean_std_scaling"]] = None
    compute_scaling_factors: bool = False
    caching: bool = False
    deterministic: bool = False
    gpu_preprocessing: bool = True
    gpu_output: bool = True

    def __post_init__(self):
        # Ensure data_path is a Path object:
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        self.data_path = self.data_path.expanduser()

        if not self.data_path.exists():
            raise ValueError(f"Path {self.data_path} does not exist")

        if not self.data_path.is_dir():
            raise ValueError(f"Path {self.data_path} is not a directory")

        # Object if caching settings are impossible:
        if self.caching:
            if self.sampling:
                raise ValueError("Sampling should be False for caching")
            if self.compute_scaling_factors:
                raise ValueError("Compute scaling factors should be False for caching")
            if self.resample_surfaces:
                raise ValueError("Resample surface should be False for caching")

        if self.phase not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError(
                f"phase should be one of ['train', 'val', 'test'], got {self.phase}"
            )
        if self.scaling_type is not None:
            if self.scaling_type not in [
                "min_max_scaling",
                "mean_std_scaling",
            ]:
                raise ValueError(
                    f"scaling_type should be one of ['min_max_scaling', 'mean_std_scaling'], got {self.scaling_type}"
                )


##### TODO
# - check the bounding box protocol works


class DoMINODataPipe(Dataset):
    """
    Datapipe for DoMINO

    Leverages a dataset for the actual reading of the data, and this
    object is responsible for preprocessing the data.

    """

    def __init__(
        self,
        input_path,
        model_type: Literal["surface", "volume", "combined"],
        **data_config_overrides,
    ):
        # Perform config packaging and validation
        self.config = DoMINODataConfig(data_path=input_path, **data_config_overrides)

        # Set up the distributed manager:
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        dist = DistributedManager()
        if self.config.gpu_preprocessing or self.config.gpu_output:
            # Make sure we move data to the right device:
            target_device = dist.device
            self.preprocess_stream = torch.cuda.Stream()
        else:
            target_device = torch.device("cpu")
            self.preprocess_stream = None

        self.device = target_device

        self.model_type = model_type

        # Update the arrays for bounding boxes:
        if hasattr(self.config.bounding_box_dims, "max") and hasattr(
            self.config.bounding_box_dims, "min"
        ):
            self.config.bounding_box_dims = [
                torch.tensor(
                    self.config.bounding_box_dims.max,
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    self.config.bounding_box_dims.min,
                    device=self.device,
                    dtype=torch.float32,
                ),
            ]
            self.volume_grid = create_grid(
                self.config.bounding_box_dims[0],
                self.config.bounding_box_dims[1],
                self.config.grid_resolution,
            )

        if hasattr(self.config.bounding_box_dims_surf, "max") and hasattr(
            self.config.bounding_box_dims_surf, "min"
        ):
            self.config.bounding_box_dims_surf = [
                torch.tensor(
                    self.config.bounding_box_dims_surf.max,
                    device=self.device,
                    dtype=torch.float32,
                ),
                torch.tensor(
                    self.config.bounding_box_dims_surf.min,
                    device=self.device,
                    dtype=torch.float32,
                ),
            ]

            self.surf_grid = create_grid(
                self.config.bounding_box_dims_surf[0],
                self.config.bounding_box_dims_surf[1],
                self.config.grid_resolution,
            )

        # Ensure the volume and surface scaling factors are torch tensors
        # and on the right device:
        if self.config.volume_factors is not None:
            self.config.volume_factors = torch.tensor(
                self.config.volume_factors, device=self.device, dtype=torch.float32
            )
        if self.config.surface_factors is not None:
            self.config.surface_factors = torch.tensor(
                self.config.surface_factors, device=self.device, dtype=torch.float32
            )

        # Always read these keys:
        self.keys_to_read = ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]

        self.keys_to_read_if_available = {
            "global_params_values": torch.tensor([30.0, 1.226], device=self.device),
            "global_params_reference": torch.tensor([30.0, 1.226], device=self.device),
        }

        self.volume_keys = ["volume_mesh_centers", "volume_fields"]
        self.surface_keys = [
            "surface_mesh_centers",
            "surface_normals",
            "surface_areas",
            "surface_fields",
        ]

        if self.model_type == "volume" or self.model_type == "combined":
            self.keys_to_read.extend(self.volume_keys)
        if self.model_type == "surface" or self.model_type == "combined":
            self.keys_to_read.extend(self.surface_keys)

        self.dataset = DrivaerMLDataset(
            data_dir=self.config.data_path,
            keys_to_read=self.keys_to_read,
            output_device=self.device,
            consumer_stream=self.preprocess_stream,
        )

        # This is thread storage for data preprocessing:
        self._preprocess_queue = {}
        self._preprocess_events = {}
        self.preprocess_depth = 2
        self.preprocess_executor = ThreadPoolExecutor(max_workers=1)

    def set_indices(self, indices: list[int]):
        """
        Set the indices for the dataset for this epoch.
        """

        # TODO - this needs to block while anything is in the preprocess queue.

        self.indices = indices

    def __len__(self):
        return len(self.dataset)

    @torch.compile(dynamic=True)
    def compute_stl_scaling(
        self, stl_vertices: torch.Tensor, bounding_box_dims_surf: torch.Tensor | None
    ):
        """
        Compute the min and max for the defining mesh.

        """

        s_min = torch.amin(stl_vertices, 0)
        s_max = torch.amax(stl_vertices, 0)

        length_scale = torch.amax(s_max - s_min)

        # if dynamic_bbox_scaling:
        # Check the bounding box is not unit length

        if bounding_box_dims_surf is not None:
            s_max = bounding_box_dims_surf[0]
            s_min = bounding_box_dims_surf[1]
            surf_grid = self.surf_grid
        else:
            # Create the grid:
            surf_grid = create_grid(s_max, s_min, self.grid_resolution)

        surf_grid_max_min = torch.stack([s_min, s_max])

        return s_min, s_max, length_scale, surf_grid_max_min, surf_grid

    @profile
    def process_combined(
        self,
        s_min,
        s_max,
        surf_grid,
        stl_vertices,
        mesh_indices_flattened,
    ):
        # SDF calculation on the grid using WARP
        nx, ny, nz = self.config.grid_resolution

        sdf_surf_grid, _ = signed_distance_field(
            stl_vertices,
            mesh_indices_flattened,
            surf_grid,
            use_sign_winding_number=True,
        )
        sdf_surf_grid = surf_grid

        if self.config.sampling:
            geometry_points = self.config.geom_points_sample
            geometry_coordinates_sampled, idx_geometry = shuffle_array(
                stl_vertices, geometry_points
            )
            if geometry_coordinates_sampled.shape[0] < geometry_points:
                geometry_coordinates_sampled = pad(
                    geometry_coordinates_sampled, geometry_points, pad_value=-100.0
                )
            geom_centers = geometry_coordinates_sampled
        else:
            geom_centers = stl_vertices

        return (sdf_surf_grid, geom_centers)

    @torch.compile(dynamic=True)
    def process_surface(
        self,
        s_min: torch.Tensor,
        s_max: torch.Tensor,
        center_of_mass: torch.Tensor,
        surf_grid: torch.Tensor,
        surface_coordinates: torch.Tensor,
        surface_normals: torch.Tensor,
        surface_sizes: torch.Tensor,
        surface_fields: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        nx, ny, nz = self.config.grid_resolution

        return_dict = {}

        # Remove any sizes <= 0:
        idx = surface_sizes > 0
        surface_sizes = surface_sizes[idx]
        surface_fields = surface_fields[idx]
        surface_normals = surface_normals[idx]
        surface_coordinates = surface_coordinates[idx]

        if self.config.resample_surfaces:
            if self.config.resampling_points > surface_coordinates.shape[0]:
                resampling_points = surface_coordinates.shape[0]
            else:
                resampling_points = self.config.resampling_points

            surface_coordinates, idx_s = shuffle_array(
                surface_coordinates, resampling_points
            )
            surface_normals = surface_normals[idx_s]
            surface_sizes = surface_sizes[idx_s]
            surface_fields = surface_fields[idx_s]

        c_max = self.config.bounding_box_dims[0]
        c_min = self.config.bounding_box_dims[1]

        if self.config.sample_in_bbox:
            ids_min = surface_coordinates[:] > c_min
            ids_max = surface_coordinates[:] < c_max

            ids_in_bbox = ids_min & ids_max
            ids_in_bbox = ids_in_bbox.all(dim=-1)

            surface_coordinates = surface_coordinates[ids_in_bbox]
            surface_normals = surface_normals[ids_in_bbox]
            surface_sizes = surface_sizes[ids_in_bbox]
            surface_fields = surface_fields[ids_in_bbox]

        # Compute the positional encoding before sampling
        if self.config.positional_encoding:
            dx, dy, dz = (
                (s_max[0] - s_min[0]) / nx,
                (s_max[1] - s_min[1]) / ny,
                (s_max[2] - s_min[2]) / nz,
            )
            pos_normals_com_surface = calculate_normal_positional_encoding(
                surface_coordinates, center_of_mass, cell_length=[dx, dy, dz]
            )
        else:
            pos_normals_com_surface = surface_coordinates - center_of_mass

        if self.config.sampling:
            # Perform the down sampling:

            if self.config.surface_sampling_algorithm == "area_weighted":
                weights = surface_sizes

            else:
                weights = None

            surface_coordinates_sampled, idx_surface = shuffle_array(
                surface_coordinates,
                self.config.surface_points_sample,
                weights=weights,
            )

            if surface_coordinates_sampled.shape[0] < self.config.surface_points_sample:
                surface_coordinates_sampled = pad(
                    surface_coordinates_sampled,
                    self.config.surface_points_sample,
                    pad_value=-10.0,
                )

            # Select out the sampled points for non-neighbor arrays:
            surface_fields = surface_fields[idx_surface]
            pos_normals_com_surface = pos_normals_com_surface[idx_surface]

            # Now, perform the kNN on the sampled points:
            if self.config.num_surface_neighbors > 1:
                neighbor_indices, neighbor_distances = knn(
                    points=surface_coordinates,
                    queries=surface_coordinates_sampled,
                    k=self.config.num_surface_neighbors,
                )

                # Pull out the neighbor elements.  Note that ii is the index into the original
                # points - but only exists for the sampled points
                # In other words, a point from `surface_coordinates_sampled` has neighbors
                # from the full `surface_coordinates` array.
                surface_neighbors = surface_coordinates[neighbor_indices][:, 1:]
                surface_neighbors_normals = surface_normals[neighbor_indices][:, 1:]
                surface_neighbors_sizes = surface_sizes[neighbor_indices][:, 1:]
            else:
                surface_neighbors = surface_coordinates
                surface_neighbors_normals = surface_normals
                surface_neighbors_sizes = surface_sizes

            # Subsample the normals and sizes:
            surface_normals = surface_normals[idx_surface]
            surface_sizes = surface_sizes[idx_surface]

            # Update the coordinates to the sampled points:
            surface_coordinates = surface_coordinates_sampled

        else:
            neighbor_indices, _ = knn(
                points=surface_coordinates,
                queries=surface_coordinates,
                k=self.config.num_surface_neighbors,
            )

            # Construct the neighbors arrays:
            surface_neighbors = surface_coordinates[neighbor_indices][:, 1:]
            surface_neighbors_normals = surface_normals[neighbor_indices][:, 1:]
            surface_neighbors_sizes = surface_sizes[neighbor_indices][:, 1:]

        # Have to normalize neighbors after the kNN and sampling
        if self.config.normalize_coordinates:
            surf_grid = normalize(surf_grid, s_max, s_min)
            surface_coordinates = normalize(surface_coordinates, s_max, s_min)
            surface_neighbors = normalize(surface_neighbors, s_max, s_min)

        if self.config.scaling_type is not None:
            if self.config.surface_factors is not None:
                if self.config.scaling_type == "mean_std_scaling":
                    surf_mean = self.config.surface_factors[0]
                    surf_std = self.config.surface_factors[1]
                    # TODO - Are these array calls needed?
                    surface_fields = standardize(surface_fields, surf_mean, surf_std)
                elif self.config.scaling_type == "min_max_scaling":
                    surf_min = self.config.surface_factors[1]
                    surf_max = self.config.surface_factors[0]
                    # TODO - Are these array calls needed?
                    surface_fields = normalize(surface_fields, surf_max, surf_min)

        return_dict.update(
            {
                "pos_surface_center_of_mass": pos_normals_com_surface,
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
                "surface_fields": surface_fields,
            }
        )

        return return_dict

    @torch.compile(dynamic=True)
    def process_volume(
        self,
        s_min: torch.Tensor,
        s_max: torch.Tensor,
        volume_coordinates: torch.Tensor,
        volume_fields: torch.Tensor,
        stl_vertices: torch.Tensor,
        mesh_indices_flattened: torch.Tensor,
        center_of_mass: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return_dict = {}

        nx, ny, nz = self.config.grid_resolution

        # Determine the volume min / max locations
        if self.config.bounding_box_dims is None:
            c_max = s_max + (s_max - s_min) / 2
            c_min = s_min - (s_max - s_min) / 2
            c_min[2] = s_min[2]
        else:
            c_max = self.config.bounding_box_dims[0]
            c_min = self.config.bounding_box_dims[1]

        if self.config.sample_in_bbox:
            # Remove points in the volume that are outside
            # of the bbox area.
            min_check = volume_coordinates[:] > c_min
            max_check = volume_coordinates[:] < c_max

            ids_in_bbox = min_check & max_check
            ids_in_bbox = ids_in_bbox.all(dim=1)

            volume_coordinates = volume_coordinates[ids_in_bbox]
            volume_fields = volume_fields[ids_in_bbox]

        dx, dy, dz = (
            (c_max[0] - c_min[0]) / nx,
            (c_max[1] - c_min[1]) / ny,
            (c_max[2] - c_min[2]) / nz,
        )

        # TODO - we need to make sure if the bbox is dynamic,
        # the bounds on the grid are correct

        # # Generate a grid of specified resolution to map the bounding box
        # # The grid is used for capturing structured geometry features and SDF representation of geometry
        # grid = create_grid(c_max, c_min, [nx, ny, nz])
        # grid_reshaped = grid.reshape(nx * ny * nz, 3)

        # SDF calculation on the volume grid using WARP
        sdf_grid, _ = signed_distance_field(
            stl_vertices,
            mesh_indices_flattened,
            self.volume_grid,
            use_sign_winding_number=True,
        )

        if self.config.sampling:
            # Generate a series of idx to sample the volume
            # without replacement

            volume_coordinates_sampled, idx_volume = shuffle_array(
                volume_coordinates, self.config.volume_points_sample
            )
            volume_coordinates_sampled = volume_coordinates[idx_volume]

            if volume_coordinates_sampled.shape[0] < self.config.volume_points_sample:
                padding_size = (
                    self.config.volume_points_sample
                    - volume_coordinates_sampled.shape[0]
                )
                volume_coordinates_sampled = torch.nn.functional.pad(
                    volume_coordinates_sampled,
                    (0, 0, 0, 0, 0, padding_size),
                    mode="constant",
                    value=-10.0,
                )

            volume_fields = volume_fields[idx_volume]
            volume_coordinates = volume_coordinates_sampled

        # Get the SDF of all the selected volume coordinates,
        # And keep the closest point to each one.
        sdf_nodes, sdf_node_closest_point = signed_distance_field(
            stl_vertices,
            mesh_indices_flattened,
            volume_coordinates,
            use_sign_winding_number=True,
        )

        if self.config.positional_encoding:
            pos_normals_closest_vol = calculate_normal_positional_encoding(
                volume_coordinates,
                sdf_node_closest_point,
                cell_length=[dx, dy, dz],
            )
            pos_normals_com_vol = calculate_normal_positional_encoding(
                volume_coordinates, center_of_mass, cell_length=[dx, dy, dz]
            )
        else:
            pos_normals_closest_vol = volume_coordinates - sdf_node_closest_point
            pos_normals_com_vol = volume_coordinates - center_of_mass

        if self.config.normalize_coordinates:
            volume_coordinates = normalize(volume_coordinates, c_max, c_min)
            grid = normalize(self.volume_grid, c_max, c_min)

        if self.config.scaling_type is not None:
            if self.config.volume_factors is not None:
                if self.config.scaling_type == "mean_std_scaling":
                    vol_mean = self.config.volume_factors[0]
                    vol_std = self.config.volume_factors[1]
                    volume_fields = standardize(volume_fields, vol_mean, vol_std)
                elif self.config.scaling_type == "min_max_scaling":
                    vol_min = self.config.volume_factors[1]
                    vol_max = self.config.volume_factors[0]
                    volume_fields = normalize(volume_fields, vol_max, vol_min)

        vol_grid_max_min = torch.stack([c_min, c_max])

        return_dict.update(
            {
                "pos_volume_closest": pos_normals_closest_vol,
                "pos_volume_center_of_mass": pos_normals_com_vol,
                "grid": grid,
                "sdf_grid": sdf_grid,
                "sdf_nodes": sdf_nodes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "volume_min_max": vol_grid_max_min,
            }
        )

        return return_dict

    @profile
    def process_data(self, data_dict, idx: int):
        for key in self.keys_to_read_if_available.keys():
            if key not in data_dict:
                data_dict[key] = self.keys_to_read_if_available[key]

        with torch.cuda.stream(self.preprocess_stream):
            if self.config.deterministic:
                torch.manual_seed(idx)

            # Start building the preprocessed return dict:
            return_dict = {
                "global_params_values": data_dict["global_params_values"],
                "global_params_reference": data_dict["global_params_reference"],
            }

            # This function gets information about the surface scale,
            # and decides what the surface grid will be:
            (s_min, s_max, length_scale, surf_grid_max_min, surf_grid) = (
                self.compute_stl_scaling(
                    data_dict["stl_coordinates"], self.config.bounding_box_dims_surf
                )
            )

            # This is a center of mass computation for the stl surface,
            # using the size of each mesh point as weight.

            center_of_mass = calculate_center_of_mass(
                data_dict["stl_centers"], data_dict["stl_areas"]
            )

            # For SDF calculations, make sure the mesh_indices_flattened is an integer array:
            mesh_indices_flattened = data_dict["stl_faces"].to(torch.int32)

            return_dict.update(
                {
                    "length_scale": length_scale,
                    "surface_min_max": surf_grid_max_min,
                }
            )

            # This will compute the sdf on the surface grid and apply downsampling if needed
            sdf_surf_grid, geom_centers = self.process_combined(
                s_min,
                s_max,
                surf_grid,
                stl_vertices=data_dict["stl_coordinates"],
                mesh_indices_flattened=mesh_indices_flattened,
            )
            return_dict["surf_grid"] = surf_grid
            return_dict["sdf_surf_grid"] = sdf_surf_grid
            return_dict["geometry_coordinates"] = geom_centers

            # Up to here works all in torch!

            if self.model_type == "volume" or self.model_type == "combined":
                volume_dict = self.process_volume(
                    s_min,
                    s_max,
                    volume_coordinates=data_dict["volume_mesh_centers"],
                    volume_fields=data_dict["volume_fields"],
                    stl_vertices=data_dict["stl_coordinates"],
                    mesh_indices_flattened=mesh_indices_flattened,
                    center_of_mass=center_of_mass,
                )

                return_dict.update(volume_dict)

            if self.model_type == "surface" or self.model_type == "combined":
                surface_dict = self.process_surface(
                    s_min,
                    s_max,
                    center_of_mass,
                    surf_grid,
                    surface_coordinates=data_dict["surface_mesh_centers"],
                    surface_normals=data_dict["surface_normals"],
                    surface_sizes=data_dict["surface_areas"],
                    surface_fields=data_dict["surface_fields"],
                )
                return_dict.update(surface_dict)

            if self.device.type == "cuda":
                self._preprocess_events[idx] = torch.cuda.Event()
                self._preprocess_events[idx].record(self.preprocess_stream)

            # Mark all cuda tensors to be consumed on the main stream:
            if self.device.type == "cuda":
                for key in return_dict.keys():
                    if isinstance(return_dict[key], torch.Tensor):
                        return_dict[key].record_stream(torch.cuda.default_stream())

        return return_dict

    @profile
    def __getitem__(self, idx):
        """
        Function for fetching and processing a single file's data.

        Domino, in general, expects one example per file and the files
        are relatively large due to the mesh size.
        """

        index = self.idx_to_index(idx)

        # Get the preprocessed data:
        data_dict = self.get_preprocessed(idx)
        if data_dict is None:
            # If no preprocessing was done for this index, process it now

            # Get the data from the dataset.
            # Under the hood, this may be fetching preloaded data.
            data_dict = self.dataset[index]
            data_dict = self.process_data(data_dict, idx)

        # This blocks the main stream until the preprocessing has transferred to GPU
        if idx in self._preprocess_events:
            torch.cuda.current_stream().wait_event(self._preprocess_events[idx])
            self._preprocess_events.pop(idx)

        return data_dict

    def idx_to_index(self, idx):
        if hasattr(self, "indices"):
            return self.indices[idx]

        return idx

    def preprocess(self, idx: int) -> None:
        """
        Start preprocessing for the given index (1 step ahead).
        This processes preloaded data or loads it if not available.
        """
        if idx in self._preprocess_queue:
            # Skip items that are already being preprocessed
            return

        def _preprocess_worker():
            index = self.idx_to_index(idx)
            # Try to get preloaded data first
            data_dict = self.dataset[index]
            # Process the data
            return self.process_data(data_dict, idx)

        # Submit preprocessing task to thread pool
        self._preprocess_queue[idx] = self.preprocess_executor.submit(
            _preprocess_worker
        )

    def get_preprocessed(self, idx: int) -> dict | None:
        """
        Retrieve preprocessed data (blocking if not ready).
        Returns None if no preprocessing is in progress for this index.
        """
        if idx not in self._preprocess_queue:
            return None

        result = self._preprocess_queue[idx].result()  # Block until ready
        self._preprocess_queue.pop(idx)  # Clear after getting result

        return result

    def __next__(self):
        # To iterate through the data efficiently, he have to implement the
        # following, assuming a steady state

        # - start the dataset loading at idx + 2
        # - start the preprocessing pipe at idx + 1
        #   - the preprocessing pipe has to implicitly wait for idx +1 in the dataset
        # - wait for the preprocessing pipe at idx to finish
        # return the data.
        if self.i >= len(self.dataset):
            self.i = 0
            raise StopIteration

        current_idx = self.i

        # Start loading two ahead:
        if len(self.dataset) >= current_idx + 2:
            self.dataset.preload(self.idx_to_index(current_idx + 2))

        # Start preprocessing one ahead:
        if len(self.dataset) >= current_idx + 1:
            self.preprocess(current_idx + 1)

        # If no preprocessing was done for this index, process it now
        data = self.__getitem__(current_idx)

        self.i += 1
        return data

    def __iter__(self):
        # When starting the iterator method, start loading the data
        # at idx = 0, idx = 1
        # Start preprocessing at idx = 0, when the load completes

        self.i = 0

        # Trigger the dataset to start loading index 0:
        if len(self.dataset) >= 1:
            self.dataset.preload(self.idx_to_index(self.i))
        if len(self.dataset) >= 2:
            self.dataset.preload(self.idx_to_index(self.i + 1))

        # Start preprocessing index 0
        self.preprocess(self.i)

        return self


@profile
def compute_scaling_factors(cfg: DictConfig, input_path: str, use_cache: bool) -> None:
    # Create a dataset for just the field keys:

    dataset = DrivaerMLDataset(
        data_dir=input_path,
        keys_to_read=["volume_fields", "surface_fields"],
        output_device=torch.device("cuda"),  # TODO - configure this more carefully here
    )

    mean, std, min_val, max_val = compute_mean_std_min_max(
        dataset,
        field_keys=["volume_fields", "surface_fields"],
    )

    return mean, std, min_val, max_val


class CachedDoMINODataset(Dataset):
    """
    Dataset for reading cached DoMINO data files, with optional resampling.
    Acts as a drop-in replacement for DoMINODataPipe.
    """

    # @nvtx_annotate(message="CachedDoMINODataset __init__")
    def __init__(
        self,
        data_path: Union[str, Path],
        phase: Literal["train", "val", "test"] = "train",
        sampling: bool = False,
        volume_points_sample: Optional[int] = None,
        surface_points_sample: Optional[int] = None,
        geom_points_sample: Optional[int] = None,
        model_type=None,  # Model_type, surface, volume or combined
        deterministic_seed=False,
        surface_sampling_algorithm="area_weighted",
    ):
        super().__init__()

        self.model_type = model_type
        if deterministic_seed:
            np.random.seed(42)

        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path.expanduser()

        if not self.data_path.exists():
            raise AssertionError(f"Path {self.data_path} does not exist")
        if not self.data_path.is_dir():
            raise AssertionError(f"Path {self.data_path} is not a directory")

        self.deterministic_seed = deterministic_seed
        self.sampling = sampling
        self.volume_points = volume_points_sample
        self.surface_points = surface_points_sample
        self.geom_points = geom_points_sample
        self.surface_sampling_algorithm = surface_sampling_algorithm

        self.filenames = get_filenames(self.data_path, exclude_dirs=True)

        total_files = len(self.filenames)

        self.phase = phase
        self.indices = np.array(range(total_files))

        np.random.shuffle(self.indices)

        if not self.filenames:
            raise AssertionError(f"No cached files found in {self.data_path}")

    def __len__(self):
        return len(self.indices)

    # @nvtx_annotate(message="CachedDoMINODataset __getitem__")
    def __getitem__(self, idx):
        if self.deterministic_seed:
            np.random.seed(idx)
        nvtx.range_push("Load cached file")

        index = self.indices[idx]
        cfd_filename = self.filenames[index]

        filepath = self.data_path / cfd_filename
        result = np.load(filepath, allow_pickle=True).item()
        result = {
            k: v.numpy() if isinstance(v, Tensor) else v for k, v in result.items()
        }

        nvtx.range_pop()
        if not self.sampling:
            return result

        nvtx.range_push("Sample points")

        # Sample volume points if present
        if "volume_mesh_centers" in result and self.volume_points:
            coords_sampled, idx_volume = shuffle_array(
                result["volume_mesh_centers"], self.volume_points
            )
            if coords_sampled.shape[0] < self.volume_points:
                coords_sampled = pad(
                    coords_sampled, self.volume_points, pad_value=-10.0
                )

            result["volume_mesh_centers"] = coords_sampled
            for key in [
                "volume_fields",
                "pos_volume_closest",
                "pos_volume_center_of_mass",
                "sdf_nodes",
            ]:
                if key in result:
                    result[key] = result[key][idx_volume]

        # Sample surface points if present
        if "surface_mesh_centers" in result and self.surface_points:
            if self.surface_sampling_algorithm == "area_weighted":
                coords_sampled, idx_surface = area_weighted_shuffle_array(
                    result["surface_mesh_centers"],
                    self.surface_points,
                    result["surface_areas"],
                )
            else:
                coords_sampled, idx_surface = shuffle_array(
                    result["surface_mesh_centers"], self.surface_points
                )

            if coords_sampled.shape[0] < self.surface_points:
                coords_sampled = pad(
                    coords_sampled, self.surface_points, pad_value=-10.0
                )

            ii = result["neighbor_indices"]
            result["surface_mesh_neighbors"] = result["surface_mesh_centers"][ii]
            result["surface_neighbors_normals"] = result["surface_normals"][ii]
            result["surface_neighbors_areas"] = result["surface_areas"][ii]

            result["surface_mesh_centers"] = coords_sampled

            for key in [
                "surface_fields",
                "surface_areas",
                "surface_normals",
                "pos_surface_center_of_mass",
                "surface_mesh_neighbors",
                "surface_neighbors_normals",
                "surface_neighbors_areas",
            ]:
                if key in result:
                    result[key] = result[key][idx_surface]

            del result["neighbor_indices"]

        # Sample geometry points if present
        if "geometry_coordinates" in result and self.geom_points:
            coords_sampled, _ = shuffle_array(
                result["geometry_coordinates"], self.geom_points
            )
            if coords_sampled.shape[0] < self.geom_points:
                coords_sampled = pad(coords_sampled, self.geom_points, pad_value=-100.0)
            result["geometry_coordinates"] = coords_sampled

        nvtx.range_pop()
        return result


def create_domino_dataset(
    cfg, phase, volume_variable_names, surface_variable_names, vol_factors, surf_factors
):
    if phase == "train":
        input_path = cfg.data.input_dir
    elif phase == "val":
        input_path = cfg.data.input_dir_val
    else:
        raise ValueError(f"Invalid phase {phase}")

    if cfg.data_processor.use_cache:
        return CachedDoMINODataset(
            input_path,
            phase=phase,
            sampling=True,
            volume_points_sample=cfg.model.volume_points_sample,
            surface_points_sample=cfg.model.surface_points_sample,
            geom_points_sample=cfg.model.geom_points_sample,
            model_type=cfg.model.model_type,
            surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        )
    else:
        overrides = {}
        if hasattr(cfg.data, "gpu_preprocessing"):
            overrides["gpu_preprocessing"] = cfg.data.gpu_preprocessing

        if hasattr(cfg.data, "gpu_output"):
            overrides["gpu_output"] = cfg.data.gpu_output

        return DoMINODataPipe(
            input_path,
            phase=phase,
            grid_resolution=cfg.model.interp_res,
            volume_variables=volume_variable_names,
            surface_variables=surface_variable_names,
            normalize_coordinates=True,
            sampling=True,
            sample_in_bbox=True,
            volume_points_sample=cfg.model.volume_points_sample,
            surface_points_sample=cfg.model.surface_points_sample,
            geom_points_sample=cfg.model.geom_points_sample,
            positional_encoding=cfg.model.positional_encoding,
            volume_factors=vol_factors,
            surface_factors=surf_factors,
            scaling_type=cfg.model.normalization,
            model_type=cfg.model.model_type,
            bounding_box_dims=cfg.data.bounding_box,
            bounding_box_dims_surf=cfg.data.bounding_box_surface,
            num_surface_neighbors=cfg.model.num_neighbors_surface,
            resample_surfaces=cfg.model.resampling_surface_mesh.resample,
            resampling_points=cfg.model.resampling_surface_mesh.points,
            surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
            **overrides,
        )


if __name__ == "__main__":
    fm_data = DoMINODataPipe(
        data_path="/code/processed_data/new_models_1/",
        phase="train",
        sampling=False,
        sample_in_bbox=False,
    )
