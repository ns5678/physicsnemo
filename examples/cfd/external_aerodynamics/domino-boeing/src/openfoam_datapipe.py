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
This is the datapipe to read OpenFoam files (vtp/vtu/stl) and save them as point clouds 
in npy format. 

"""

import time, random, json, re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from physicsnemo.utils.domino.utils import *
from torch.utils.data import Dataset

PREF = 176.352
RHOREF = 1.375e-6
UINFTY = 2679.505

class BoeingPaths:
    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        return car_dir / "CRMHL_ap_F10_cf.stl"

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        return car_dir / "volume_geo_F10_AoA_4.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        return car_dir / "boundary_geo_F10_AoA_4.vtu"

class OpenFoamDataset(Dataset):
    """
    Datapipe for converting boeing dataset to npy
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        kind: Literal["boeing_data"] = "boeing_data",
        surface_variables: Optional[list] = [],
        volume_variables: Optional[list] = [],
        global_params_types: Optional[dict] = {},
        global_params_reference: Optional[dict] = {},
        sampling: bool = False,
        sample_in_bbox: bool = False,
        device: int = 0,
        model_type=None,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()

        self.data_path = data_path

        supported_kinds = ["boeing_data"]
        assert (
            kind in supported_kinds
        ), f"kind should be one of {supported_kinds}, got {kind}"
        
        if kind == "boeing_data":
            self.path_getter = BoeingPaths

        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"

        self.filenames = get_filenames(self.data_path)
        random.shuffle(self.filenames)
        self.indices = np.array(len(self.filenames))

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.global_params_types = global_params_types
        self.global_params_reference = global_params_reference
        self.kind = kind
        self.AoA = self.global_params_reference["AoA"]
        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cfd_filename = self.filenames[idx]
        car_dir = self.data_path / cfd_filename

        ## Extract AoA from volume filename
        volume_filepath = self.path_getter.volume_path(car_dir)
        volume_filename = volume_filepath.name  # e.g., "volume_geo_F10_AoA_4.vtu"
        # Parse AoA from filename (format: "..._AoA_X.vtu")
        aoa_match = re.search(r'AoA_(\d+(?:\.\d+)?)', volume_filename)
        if aoa_match:
            sample_AoA = np.float32(aoa_match.group(1))
        else:
            raise ValueError(f"Could not extract AoA from volume filename: {volume_filename}")

        ## Read geometry STL file
        stl_path = self.path_getter.geometry_path(car_dir)
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_stl.cell_centers().points)

        ## Read volume VTU file
        if self.model_type == "volume" or self.model_type == "combined":
            filepath = self.path_getter.volume_path(car_dir)
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(filepath)
            reader.Update()

            # Get the unstructured grid data
            polydata = reader.GetOutput()
            volume_coordinates, volume_fields = get_volume_data(
                polydata, self.volume_variables
            )
            volume_fields = np.concatenate(volume_fields, axis=-1)
            print('volume_fields shape: ', volume_fields.shape)
            print('volume_fields min: ', np.min(volume_fields, axis=0))
            print('volume_fields max: ', np.max(volume_fields, axis=0))
        
            volume_fields[:, 0:1] = volume_fields[:, 0:1] / PREF # avg pressure
            volume_fields[:, 1:2] = volume_fields[:, 1:2] / RHOREF # avg density
            volume_fields[:, 2:] = volume_fields[:, 2:]   / UINFTY # avg velocity
            print('volume_fields min after non-dimensionalization: ', np.min(volume_fields, axis=0))
            print('volume_fields max after non-dimensionalization: ', np.max(volume_fields, axis=0))
        else:
            volume_fields = None
            volume_coordinates = None

        ## Read surface VTP file
        if self.model_type == "surface" or self.model_type == "combined":
            surface_filepath = self.path_getter.surface_path(car_dir)
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(surface_filepath)
            reader.Update()
            polydata = reader.GetOutput()
            
            celldata_all = get_node_to_elem(polydata)
            celldata = celldata_all.GetCellData()
            surface_fields = get_fields(celldata, self.surface_variables)
            surface_fields = np.concatenate(surface_fields, axis=-1)

            mesh = pv.PolyData(polydata)
            surface_coordinates = np.array(mesh.cell_centers().points)

            surface_normals = np.array(mesh.cell_normals)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"])

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )

            # Non-dimensionalize surface fields
            surface_fields = surface_fields / PREF
        else:
            surface_fields = None
            surface_coordinates = None
            surface_normals = None
            surface_sizes = None

        ## Arrange global parameters reference in a list based on the type of the parameter
        global_params_reference_list = []
        for name, type in self.global_params_types.items():
            if type == "vector":
                global_params_reference_list.extend(self.global_params_reference[name])
            elif type == "scalar":
                global_params_reference_list.append(self.global_params_reference[name])
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
        global_params_values_list = []
        for key in self.global_params_types.keys():
            if key == "AoA":
                global_params_values_list.append(sample_AoA)
            else:
                raise ValueError(
                    f"Global parameter {key} not supported for  this dataset"
                )
        global_params_values = np.array(global_params_values_list, dtype=np.float32)


        # print('Surface fields shape: ', surface_fields.shape, 'Volume fields shape: ', volume_fields.shape)
        # print('Surface fields min: ', np.min(surface_fields), 'Surface fields max: ', np.max(surface_fields))
        print('Volume fields min: ', np.min(volume_fields), 'Volume fields max: ', np.max(volume_fields))
        print('global_params_values: ', global_params_values)
        print('global_params_reference: ', global_params_reference)
        
        return {
            "stl_coordinates": np.float32(stl_vertices),
            "stl_centers": np.float32(stl_centers),
            "stl_faces": np.float32(mesh_indices_flattened),
            "stl_areas": np.float32(stl_sizes),
            "surface_mesh_centers": None,
            "surface_normals": None,
            "surface_areas": None,
            "surface_fields": None,
            "volume_fields": np.float32(volume_fields),
            "volume_mesh_centers": np.float32(volume_coordinates),
            "filename": cfd_filename,
            "global_params_values": global_params_values,
            "global_params_reference": global_params_reference,
        }


if __name__ == "__main__":
    fm_data = OpenFoamDataset(
        data_path="/code/aerofoundationdata/",
        phase="train",
        volume_variables=[],
        surface_variables=[],
        global_params_types={},
        global_params_reference={},
        sampling=False,
        sample_in_bbox=False,
    )
    d_dict = fm_data[1]