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
This code defines a distributed pipeline for testing the DoMINO model on
CFD datasets. It includes the instantiating the DoMINO model and datapipe,
automatically loading the most recent checkpoint, reading the VTP/VTU/STL
testing files, calculation of parameters required for DoMINO model and
evaluating the model in parallel using DistributedDataParallel across multiple
GPUs. This is a common recipe that enables training of combined models for surface
and volume as well either of them separately. The model predictions are loaded in
the the VTP/VTU files and saved in the specified directory. The eval tab in
config.yaml can be used to specify the input and output directories.
"""

import os, re
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

import numpy as np
import cupy as cp

from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import pandas as pd
import pyvista as pv

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset

import vtk
from vtk.util import numpy_support

from physicsnemo.distributed import DistributedManager
from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *
from physicsnemo.utils.sdf import signed_distance_field

## Constants across simulation files for reference pressure, rho, velocity
PREF = np.float32(176.352)
RHOREF = np.float32(1.375e-6)
UINFTY = np.float32(2679.505)


def loss_fn(output, target):
    masked_loss = torch.mean(((output - target) ** 2.0), (0, 1, 2))
    loss = torch.mean(masked_loss)
    return loss


def test_step(data_dict, model, device, cfg, vol_factors, surf_factors):
    avg_tloss_vol = 0.0
    avg_tloss_surf = 0.0
    running_tloss_vol = 0.0
    running_tloss_surf = 0.0

    if cfg.model.model_type == "volume" or cfg.model.model_type == "combined":
        output_features_vol = True
    else:
        output_features_vol = None

    if cfg.model.model_type == "surface" or cfg.model.model_type == "combined":
        output_features_surf = True
    else:
        output_features_surf = None

    with torch.no_grad():
        point_batch_size = 1024000
        data_dict = dict_to_device(data_dict, device)

        # Non-dimensionalization factors
        length_scale = data_dict["length_scale"]

        global_params_values = data_dict["global_params_values"]
        global_params_reference = data_dict["global_params_reference"]

        # STL nodes
        geo_centers = data_dict["geometry_coordinates"]

        # Bounding box grid
        s_grid = data_dict["surf_grid"]
        sdf_surf_grid = data_dict["sdf_surf_grid"]
        # Scaling factors
        surf_max = data_dict["surface_min_max"][:, 1]
        surf_min = data_dict["surface_min_max"][:, 0]

        if output_features_vol is not None:
            # Represent geometry on computational grid
            # Computational domain grid
            p_grid = data_dict["grid"]
            sdf_grid = data_dict["sdf_grid"]
            # Scaling factors
            vol_max = data_dict["volume_min_max"][:, 1]
            vol_min = data_dict["volume_min_max"][:, 0]

            # Normalize based on computational domain
            geo_centers_vol = 2.0 * (geo_centers - vol_min) / (vol_max - vol_min) - 1
            encoding_g_vol = model.geo_rep_volume(geo_centers_vol, p_grid, sdf_grid)

        if output_features_surf is not None:
            # Represent geometry on bounding box
            geo_centers_surf = (
                2.0 * (geo_centers - surf_min) / (surf_max - surf_min) - 1
            )
            encoding_g_surf = model.geo_rep_surface(
                geo_centers_surf, s_grid, sdf_surf_grid
            )

        if (
            output_features_vol is not None
            and output_features_surf is not None
            and cfg.model.combine_volume_surface
        ):
            encoding_g = torch.cat((encoding_g_vol, encoding_g_surf), axis=1)
            encoding_g_surf = model.combined_unet_surf(encoding_g)
            encoding_g_vol = model.combined_unet_vol(encoding_g)

        if output_features_vol is not None:
            # First calculate volume predictions if required
            volume_mesh_centers = data_dict["volume_mesh_centers"]
            target_vol = data_dict["volume_fields"]
            # SDF on volume mesh nodes
            sdf_nodes = data_dict["sdf_nodes"]
            # Positional encoding based on closest point on surface to a volume node
            pos_volume_closest = data_dict["pos_volume_closest"]
            # Positional encoding based on center of mass of geometry to volume node
            pos_volume_center_of_mass = data_dict["pos_volume_center_of_mass"]
            p_grid = data_dict["grid"]

            prediction_vol = np.zeros_like(target_vol.cpu().numpy())
            num_points = volume_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))

            start_time = time.time()

            for p in range(subdomain_points + 1):
                start_idx = p * point_batch_size
                end_idx = (p + 1) * point_batch_size
                with torch.no_grad():
                    target_batch = target_vol[:, start_idx:end_idx]
                    volume_mesh_centers_batch = volume_mesh_centers[
                        :, start_idx:end_idx
                    ]
                    sdf_nodes_batch = sdf_nodes[:, start_idx:end_idx]
                    pos_volume_closest_batch = pos_volume_closest[:, start_idx:end_idx]
                    pos_normals_com_batch = pos_volume_center_of_mass[
                        :, start_idx:end_idx
                    ]
                    geo_encoding_local = model.geo_encoding_local(
                        0.5 * encoding_g_vol,
                        volume_mesh_centers_batch,
                        p_grid,
                        mode="volume",
                    )
                    if cfg.model.use_sdf_in_basis_func:
                        pos_encoding = torch.cat(
                            (
                                sdf_nodes_batch,
                                pos_volume_closest_batch,
                                pos_normals_com_batch,
                            ),
                            axis=-1,
                        )
                    else:
                        pos_encoding = pos_normals_com_batch
                    pos_encoding = model.position_encoder(
                        pos_encoding, eval_mode="volume"
                    )
                    tpredictions_batch = model.calculate_solution(
                        volume_mesh_centers_batch,
                        geo_encoding_local,
                        pos_encoding,
                        global_params_values,
                        global_params_reference,
                        num_sample_points=cfg.model.num_neighbors_volume,
                        eval_mode="volume",
                    )
                    running_tloss_vol += loss_fn(tpredictions_batch, target_batch)
                    prediction_vol[:, start_idx:end_idx] = (
                        tpredictions_batch.cpu().numpy()
                    )

            prediction_vol = unnormalize(prediction_vol, vol_factors[0], vol_factors[1])

            prediction_vol[:, :, :1] = prediction_vol[:, :, 0:1] * PREF
            prediction_vol[:, :, 1:] = prediction_vol[:, :, 1:] * UINFTY

        else:
            prediction_vol = None

        if output_features_surf is not None:
            # Next calculate surface predictions
            # Sampled points on surface
            surface_mesh_centers = data_dict["surface_mesh_centers"]
            surface_normals = data_dict["surface_normals"]
            surface_areas = data_dict["surface_areas"]

            # Neighbors of sampled points on surface
            surface_mesh_neighbors = data_dict["surface_mesh_neighbors"]
            surface_neighbors_normals = data_dict["surface_neighbors_normals"]
            surface_neighbors_areas = data_dict["surface_neighbors_areas"]
            surface_areas = torch.unsqueeze(surface_areas, -1)
            surface_neighbors_areas = torch.unsqueeze(surface_neighbors_areas, -1)
            pos_surface_center_of_mass = data_dict["pos_surface_center_of_mass"]
            num_points = surface_mesh_centers.shape[1]
            subdomain_points = int(np.floor(num_points / point_batch_size))

            target_surf = data_dict["surface_fields"]
            prediction_surf = np.zeros_like(target_surf.cpu().numpy())

            start_time = time.time()

            for p in range(subdomain_points + 1):
                start_idx = p * point_batch_size
                end_idx = (p + 1) * point_batch_size
                with torch.no_grad():
                    target_batch = target_surf[:, start_idx:end_idx]
                    surface_mesh_centers_batch = surface_mesh_centers[
                        :, start_idx:end_idx
                    ]
                    surface_mesh_neighbors_batch = surface_mesh_neighbors[
                        :, start_idx:end_idx
                    ]
                    surface_normals_batch = surface_normals[:, start_idx:end_idx]
                    surface_neighbors_normals_batch = surface_neighbors_normals[
                        :, start_idx:end_idx
                    ]
                    surface_areas_batch = surface_areas[:, start_idx:end_idx]
                    surface_neighbors_areas_batch = surface_neighbors_areas[
                        :, start_idx:end_idx
                    ]
                    pos_surface_center_of_mass_batch = pos_surface_center_of_mass[
                        :, start_idx:end_idx
                    ]
                    geo_encoding_local = model.geo_encoding_local(
                        0.5 * encoding_g_surf,
                        surface_mesh_centers_batch,
                        s_grid,
                        mode="surface",
                    )
                    pos_encoding = pos_surface_center_of_mass_batch
                    pos_encoding = model.position_encoder(
                        pos_encoding, eval_mode="surface"
                    )
                    tpredictions_batch = model.calculate_solution_with_neighbors(
                        surface_mesh_centers_batch,
                        geo_encoding_local,
                        pos_encoding,
                        surface_mesh_neighbors_batch,
                        surface_normals_batch,
                        surface_neighbors_normals_batch,
                        surface_areas_batch,
                        surface_neighbors_areas_batch,
                        global_params_values,
                        global_params_reference,
                        num_sample_points=cfg.model.num_neighbors_surface,
                    )
                    running_tloss_surf += loss_fn(tpredictions_batch, target_batch)
                    prediction_surf[:, start_idx:end_idx] = (
                        tpredictions_batch.cpu().numpy()
                    )

            prediction_surf = unnormalize(
                prediction_surf, surf_factors[0], surf_factors[1]
            )

            prediction_surf *= PREF

        else:
            prediction_surf = None

    return prediction_vol, prediction_surf


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    input_path = cfg.eval.test_path

    model_type = cfg.model.model_type

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        num_vol_vars = 0
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    global_features = 0
    global_params_names = list(cfg.variables.global_parameters.keys())
    for param in global_params_names:
        if cfg.variables.global_parameters[param].type == "vector":
            global_features += len(cfg.variables.global_parameters[param].reference)
        else:
            global_features += 1

    vol_save_path = os.path.join(
        cfg.eval.scaling_param_path, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        cfg.eval.scaling_param_path, "surface_scaling_factors.npy"
    )
    if os.path.exists(vol_save_path):
        vol_factors = np.load(vol_save_path)
    else:
        vol_factors = None

    if os.path.exists(surf_save_path):
        surf_factors = np.load(surf_save_path)
    else:
        surf_factors = None

    print("Vol factors:", vol_factors)
    print("Surf factors:", surf_factors)

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=global_features,
        model_parameters=cfg.model,
    ).to(dist.device)

    model = torch.compile(model, disable=True)

    checkpoint = torch.load(
        to_absolute_path(os.path.join(cfg.resume_dir, cfg.eval.checkpoint_name)),
        map_location=dist.device,
    )

    model.load_state_dict(checkpoint)

    print("Model loaded")

    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            output_device=dist.device,
            broadcast_buffers=dist.broadcast_buffers,
            find_unused_parameters=dist.find_unused_parameters,
            gradient_as_bucket_view=True,
            static_graph=True,
        )
        model = model.module

    dirnames = get_filenames(input_path)
    dev_id = torch.cuda.current_device()
    num_files = int(len(dirnames) / dist.world_size)
    dirnames_per_gpu = dirnames[int(num_files * dev_id) : int(num_files * (dev_id + 1))]

    pred_save_path = cfg.eval.save_path

    if dist.rank == 0:
        create_directory(pred_save_path)

    l2_surface_all = []
    l2_volume_all = []
    aero_forces_all = []
    for count, dirname in enumerate(dirnames_per_gpu):
        filepath = os.path.join(input_path, dirname)
        tag = re.findall(r"(LHC\d+_AoA_\d+)", dirname)[0]
        stl_path = os.path.join(filepath, f"{dirname}.stl")
        vtp_path = os.path.join(filepath, f"boundary_{dirname}.vtu")
        vtu_path = os.path.join(filepath, f"volume_{dirname}.vtu")

        vtp_pred_save_path = os.path.join(
            pred_save_path, f"boundary_{tag}_predicted.vtu"
        )
        vtu_pred_save_path = os.path.join(pred_save_path, f"volume_{tag}_predicted.vtu")

        ## Read STL file
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        length_scale = np.amax(np.amax(stl_vertices, 0) - np.amin(stl_vertices, 0))
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"], dtype=np.float32)
        stl_centers = np.array(mesh_stl.cell_centers().points, dtype=np.float32)

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        if cfg.data.bounding_box_surface is None:
            s_max = np.amax(stl_vertices, 0)
            s_min = np.amin(stl_vertices, 0)
        else:
            bounding_box_dims_surf = []
            bounding_box_dims_surf.append(np.asarray(cfg.data.bounding_box_surface.max))
            bounding_box_dims_surf.append(np.asarray(cfg.data.bounding_box_surface.min))
            s_max = np.float32(bounding_box_dims_surf[0])
            s_min = np.float32(bounding_box_dims_surf[1])

        nx, ny, nz = cfg.model.interp_res

        surf_grid = create_grid(s_max, s_min, [nx, ny, nz])
        surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)

        # SDF calculation on the grid using WARP
        sdf_surf_grid = signed_distance_field(
            cp.asarray(stl_vertices).astype(cp.float32),
            cp.asarray(mesh_indices_flattened).astype(cp.int32),
            cp.asarray(surf_grid_reshaped).astype(cp.float32),
            use_sign_winding_number=True,
            return_cupy=False,
        ).reshape(nx, ny, nz)

        surf_grid = np.float32(surf_grid)
        sdf_surf_grid = np.float32(sdf_surf_grid)
        surf_grid_max_min = np.float32(np.asarray([s_min, s_max]))

        # Get global parameters and global parameters scaling from config.yaml
        global_params_names = list(cfg.variables.global_parameters.keys())
        global_params_reference = {
            name: cfg.variables.global_parameters[name]["reference"]
            for name in global_params_names
        }
        global_params_types = {
            name: cfg.variables.global_parameters[name]["type"]
            for name in global_params_names
        }

        # Arrange global parameters reference in a list, ensuring it is flat
        global_params_reference_list = []
        for name, type in global_params_types.items():
            if type == "vector":
                global_params_reference_list.extend(global_params_reference[name])
            elif type == "scalar":
                global_params_reference_list.append(global_params_reference[name])
            else:
                raise ValueError(
                    f"Global parameter {name} not supported for  this dataset"
                )
        global_params_reference = np.array(
            global_params_reference_list, dtype=np.float32
        )

        # Define the list of global parameter values for each simulation.
        # Note: The user must ensure that the values provided here correspond to the
        # `global_parameters` specified in `config.yaml` and that these parameters
        # exist within each simulation file.
        aoa_match = re.search(r"AoA_(\d+(?:\.\d+)?)", dirname)
        if aoa_match:
            sample_AoA = np.float32(aoa_match.group(1))
        else:
            raise ValueError(f"Could not extract AoA from folder name: {dirname}")

        global_params_values_list = []
        for key in global_params_types.keys():
            if key == "AoA":
                global_params_values_list.append(sample_AoA)
            else:
                raise ValueError(
                    f"Global parameter {key} not supported for  this dataset"
                )
        global_params_values = np.array(global_params_values_list, dtype=np.float32)

        ## Read surface VTU files
        if model_type == "surface" or model_type == "combined":
            mesh = pv.read(vtp_path)

            # Keep only the arrays specified in config and convert to float32
            # Store the arrays we want to keep (surface variables + N_BF for normals)
            arrays_to_keep = surface_variable_names + ["N_BF"]

            # Get all array names in point data
            all_arrays = [
                mesh.point_data.keys()[i] for i in range(len(mesh.point_data.keys()))
            ]

            # Remove arrays not in the keep list
            for array_name in all_arrays:
                if array_name not in arrays_to_keep:
                    mesh.point_data.pop(array_name, None)

            # Convert remaining arrays to float32
            for array_name in mesh.point_data.keys():
                mesh.point_data[array_name] = mesh.point_data[array_name].astype(
                    np.float32
                )

            # Convert points to float32
            mesh.points = mesh.points.astype(np.float32)

            # Convert to cell data for processing
            mesh = mesh.point_data_to_cell_data()

            surface_fields = []
            for name in surface_variable_names:
                surface_fields.append(np.array(mesh.cell_data[name]).astype(np.float32))
            surface_fields = np.array(surface_fields)
            surface_fields = np.stack(surface_fields, axis=-1)

            surface_normals_area = np.array(mesh.cell_data["N_BF"]).astype(np.float32)
            surface_sizes = np.linalg.norm(surface_normals_area, axis=1).astype(
                np.float32
            )
            surface_normals = np.array(mesh.cell_data["N_BF"]) / np.reshape(
                surface_sizes, (-1, 1)
            )
            surface_coordinates = mesh.cell_centers().points.astype(np.float32)

            if cfg.model.num_neighbors_surface > 1:
                interp_func = KDTree(surface_coordinates)
                dd, ii = interp_func.query(
                    surface_coordinates, k=cfg.model.num_neighbors_surface
                )

                surface_neighbors = surface_coordinates[ii]
                surface_neighbors = surface_neighbors[:, 1:]

                surface_neighbors_normals = surface_normals[ii]
                surface_neighbors_normals = surface_neighbors_normals[:, 1:]
                surface_neighbors_sizes = surface_sizes[ii]
                surface_neighbors_sizes = surface_neighbors_sizes[:, 1:]
            else:
                surface_neighbors = surface_coordinates
                surface_neighbors_normals = surface_normals
                surface_neighbors_sizes = surface_sizes

            dx, dy, dz = (
                (s_max[0] - s_min[0]) / nx,
                (s_max[1] - s_min[1]) / ny,
                (s_max[2] - s_min[2]) / nz,
            )

            if cfg.model.positional_encoding:
                pos_surface_center_of_mass = calculate_normal_positional_encoding(
                    surface_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_surface_center_of_mass = surface_coordinates - center_of_mass

            surface_coordinates = normalize(surface_coordinates, s_max, s_min)
            surface_neighbors = normalize(surface_neighbors, s_max, s_min)
            surf_grid = normalize(surf_grid, s_max, s_min)

        else:
            surface_coordinates = None
            surface_fields = None
            surface_sizes = None
            surface_normals = None
            surface_neighbors = None
            surface_neighbors_normals = None
            surface_neighbors_sizes = None
            pos_surface_center_of_mass = None

        ## Read and prune the VTU files to only have arrays in volume_variables
        if model_type == "volume" or model_type == "combined":

            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(vtu_path)
            reader.Update()
            polydata_vol = reader.GetOutput()

            # Keep only the arrays specified in config and convert to float32
            point_data = polydata_vol.GetPointData()

            # Store the arrays we want to keep in a dictionary
            arrays_to_keep = {}
            for var_name in volume_variable_names:
                if point_data.HasArray(var_name):
                    array = point_data.GetArray(var_name)
                    array_np = numpy_support.vtk_to_numpy(array).astype(np.float32)
                    arrays_to_keep[var_name] = array_np

            # Remove all arrays for point data
            num_arrays = point_data.GetNumberOfArrays()
            for i in range(num_arrays - 1, -1, -1):
                array_name = point_data.GetArray(i).GetName()
                point_data.RemoveArray(array_name)

            # Add back only the arrays we want to keep as float32
            for var_name, array_np in arrays_to_keep.items():
                array_float32 = numpy_support.numpy_to_vtk(array_np)
                array_float32.SetName(var_name)
                point_data.AddArray(array_float32)

            # Convert points to float32
            points = polydata_vol.GetPoints()
            points_np = numpy_support.vtk_to_numpy(points.GetData()).astype(np.float32)
            points_float32 = numpy_support.numpy_to_vtk(points_np)
            points.SetData(points_float32)

            # Create new polydata as point cloud (just points + point data, no cells/connectivity)
            new_polydata = vtk.vtkPolyData()
            new_polydata.SetPoints(polydata_vol.GetPoints())

            # Copy all point data arrays to the new polydata
            point_data_source = polydata_vol.GetPointData()
            point_data_dest = new_polydata.GetPointData()
            for i in range(point_data_source.GetNumberOfArrays()):
                point_data_dest.AddArray(point_data_source.GetArray(i))

            # Replace polydata_vol with point cloud version (no cells)
            polydata_vol = new_polydata

            volume_coordinates, volume_fields = get_volume_data(
                polydata_vol, volume_variable_names
            )
            volume_coordinates = np.float32(volume_coordinates)
            volume_fields = np.concatenate(volume_fields, axis=-1).astype(np.float32)

            bounding_box_dims = []
            bounding_box_dims.append(np.asarray(cfg.data.bounding_box.max))
            bounding_box_dims.append(np.asarray(cfg.data.bounding_box.min))

            v_max = np.amax(volume_coordinates, 0)
            v_min = np.amin(volume_coordinates, 0)
            if bounding_box_dims is None:
                c_max = s_max + (s_max - s_min) / 2
                c_min = s_min - (s_max - s_min) / 2
                c_min[2] = s_min[2]
            else:
                c_max = np.float32(bounding_box_dims[0])
                c_min = np.float32(bounding_box_dims[1])

            dx, dy, dz = (
                (c_max[0] - c_min[0]) / nx,
                (c_max[1] - c_min[1]) / ny,
                (c_max[2] - c_min[2]) / nz,
            )
            # Generate a grid of specified resolution to map the bounding box
            # The grid is used for capturing structured geometry features and SDF representation of geometry
            grid = create_grid(c_max, c_min, [nx, ny, nz])
            grid_reshaped = grid.reshape(nx * ny * nz, 3)

            # SDF calculation on the grid using WARP
            sdf_grid = signed_distance_field(
                cp.asarray(stl_vertices).astype(cp.float32),
                cp.asarray(mesh_indices_flattened).astype(cp.int32),
                cp.asarray(grid_reshaped).astype(cp.float32),
                use_sign_winding_number=True,
                return_cupy=False,
            ).reshape(nx, ny, nz)

            # SDF calculation
            sdf_nodes, sdf_node_closest_point = signed_distance_field(
                cp.asarray(stl_vertices).astype(cp.float32),
                cp.asarray(mesh_indices_flattened).astype(cp.int32),
                cp.asarray(volume_coordinates).astype(cp.float32),
                include_hit_points=True,
                use_sign_winding_number=True,
                return_cupy=False,
            )
            sdf_nodes = sdf_nodes.reshape(-1, 1)

            if cfg.model.positional_encoding:
                pos_volume_closest = calculate_normal_positional_encoding(
                    volume_coordinates, sdf_node_closest_point, cell_length=[dx, dy, dz]
                )
                pos_volume_center_of_mass = calculate_normal_positional_encoding(
                    volume_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_volume_closest = volume_coordinates - sdf_node_closest_point
                pos_volume_center_of_mass = volume_coordinates - center_of_mass

            volume_coordinates = normalize(volume_coordinates, c_max, c_min)
            grid = normalize(grid, c_max, c_min)
            vol_grid_max_min = np.asarray([c_min, c_max])

        else:
            volume_coordinates = None
            volume_fields = None
            pos_volume_closest = None
            pos_volume_center_of_mass = None

        geom_centers = np.float32(stl_vertices)

        if model_type == "combined":
            # Add the parameters to the dictionary
            data_dict = {
                "pos_volume_closest": pos_volume_closest,
                "pos_volume_center_of_mass": pos_volume_center_of_mass,
                "pos_surface_center_of_mass": pos_surface_center_of_mass,
                "geometry_coordinates": geom_centers,
                "grid": grid,
                "surf_grid": surf_grid,
                "sdf_grid": sdf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "sdf_nodes": sdf_nodes,
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "surface_fields": surface_fields,
                "volume_min_max": vol_grid_max_min,
                "surface_min_max": surf_grid_max_min,
                "length_scale": np.array(length_scale, dtype=np.float32),
                "global_params_values": np.expand_dims(
                    np.array(global_params_values, dtype=np.float32), -1
                ),
                "global_params_reference": np.expand_dims(
                    np.array(global_params_reference, dtype=np.float32), -1
                ),
            }
        elif model_type == "surface":
            data_dict = {
                "pos_surface_center_of_mass": np.float32(pos_surface_center_of_mass),
                "geometry_coordinates": np.float32(geom_centers),
                "surf_grid": np.float32(surf_grid),
                "sdf_surf_grid": np.float32(sdf_surf_grid),
                "surface_mesh_centers": np.float32(surface_coordinates),
                "surface_mesh_neighbors": np.float32(surface_neighbors),
                "surface_normals": np.float32(surface_normals),
                "surface_neighbors_normals": np.float32(surface_neighbors_normals),
                "surface_areas": np.float32(surface_sizes),
                "surface_neighbors_areas": np.float32(surface_neighbors_sizes),
                "surface_fields": np.float32(surface_fields),
                "surface_min_max": np.float32(surf_grid_max_min),
                "length_scale": np.array(length_scale, dtype=np.float32),
                "global_params_values": np.expand_dims(
                    np.array(global_params_values, dtype=np.float32), -1
                ),
                "global_params_reference": np.expand_dims(
                    np.array(global_params_reference, dtype=np.float32), -1
                ),
            }
        elif model_type == "volume":
            data_dict = {
                "pos_volume_closest": pos_volume_closest,
                "pos_volume_center_of_mass": pos_volume_center_of_mass,
                "geometry_coordinates": geom_centers,
                "grid": grid,
                "surf_grid": surf_grid,
                "sdf_grid": sdf_grid,
                "sdf_surf_grid": sdf_surf_grid,
                "sdf_nodes": sdf_nodes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "volume_min_max": vol_grid_max_min,
                "surface_min_max": surf_grid_max_min,
                "length_scale": np.array(length_scale, dtype=np.float32),
                "global_params_values": np.expand_dims(
                    np.array(global_params_values, dtype=np.float32), -1
                ),
                "global_params_reference": np.expand_dims(
                    np.array(global_params_reference, dtype=np.float32), -1
                ),
            }

        data_dict = {
            key: torch.from_numpy(np.expand_dims(np.float32(value), 0))
            for key, value in data_dict.items()
        }

        prediction_vol, prediction_surf = test_step(
            data_dict, model, dist.device, cfg, vol_factors, surf_factors
        )

        if prediction_surf is not None:
            surface_sizes = np.expand_dims(surface_sizes, -1)

            l2_gt = np.mean(np.square(surface_fields), (0))
            l2_error = np.mean(np.square(prediction_surf[0] - surface_fields), (0))
            l2_surface_all.append(np.sqrt(l2_error / l2_gt))

            print(
                "Surface L-2 norm:",
                dirname,
                np.sqrt(l2_error) / np.sqrt(l2_gt),
            )

        if prediction_vol is not None:
            target_vol = volume_fields
            prediction_vol = prediction_vol[0]
            c_min = vol_grid_max_min[0]
            c_max = vol_grid_max_min[1]
            volume_coordinates = unnormalize(volume_coordinates, c_max, c_min)
            ids_in_bbox = np.where(
                (volume_coordinates[:, 0] < c_min[0])
                | (volume_coordinates[:, 0] > c_max[0])
                | (volume_coordinates[:, 1] < c_min[1])
                | (volume_coordinates[:, 1] > c_max[1])
                | (volume_coordinates[:, 2] < c_min[2])
                | (volume_coordinates[:, 2] > c_max[2])
            )
            target_vol[ids_in_bbox] = 0.0
            prediction_vol[ids_in_bbox] = 0.0
            l2_gt = np.mean(np.square(target_vol), (0))
            l2_error = np.mean(np.square(prediction_vol - target_vol), (0))
            print(
                "Volume L-2 norm:",
                dirname,
                np.sqrt(l2_error) / np.sqrt(l2_gt),
            )
            l2_volume_all.append(np.sqrt(l2_error) / np.sqrt(l2_gt))

        if prediction_surf is not None:
            # Add prediction arrays to mesh cell data
            mesh[f"{surface_variable_names[0]}Pred"] = (
                prediction_surf[0, :, 0:1].astype(np.float32).flatten()
            )
            mesh[f"{surface_variable_names[1]}Pred"] = prediction_surf[0, :, 1:].astype(
                np.float32
            )

            mesh_with_point_data = mesh.cell_data_to_point_data()

            mesh_with_point_data.save(vtp_pred_save_path)

        if prediction_vol is not None:

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 0:1].astype(np.float32))
            volParam_vtk.SetName(f"{volume_variable_names[0]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            volParam_vtk = numpy_support.numpy_to_vtk(prediction_vol[:, 1:].astype(np.float32))
            volParam_vtk.SetName(f"{volume_variable_names[1]}Pred")
            polydata_vol.GetPointData().AddArray(volParam_vtk)

            # Convert polydata (point cloud) to unstructured grid for VTU format
            # VTU requires vtkUnstructuredGrid, not vtkPolyData
            unstructured_grid = vtk.vtkUnstructuredGrid()
            unstructured_grid.SetPoints(polydata_vol.GetPoints())

            # Copy all point data arrays
            point_data_source = polydata_vol.GetPointData()
            point_data_dest = unstructured_grid.GetPointData()
            for i in range(point_data_source.GetNumberOfArrays()):
                point_data_dest.AddArray(point_data_source.GetArray(i))

            write_to_vtu(unstructured_grid, vtu_pred_save_path)

    l2_surface_all = np.asarray(l2_surface_all)  # num_files, 4
    l2_volume_all = np.asarray(l2_volume_all)  # num_files, 4
    l2_surface_mean = np.mean(l2_surface_all, 0)
    l2_volume_mean = np.mean(l2_volume_all, 0)
    print(
        f"Mean over all samples, surface={l2_surface_mean} and volume={l2_volume_mean}"
    )


if __name__ == "__main__":
    main()
