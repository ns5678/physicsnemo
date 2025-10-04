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
This is the datapipe to read Cadence files from Boeing (vtp/vtu/stl) and save them as point clouds 
in npy format. 
"""

import time, random, json, re
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from physicsnemo.utils.domino.utils import *
from torch.utils.data import Dataset

## Constants across simulation files for reference pressure, rho, velocity
PREF = np.float32(176.352)
RHOREF = np.float32(1.375e-6)
UINFTY = np.float32(2679.505)

"""Class that defines the structure of the data files inside simulation folders
"""
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

## TODO: Change the name of the class to better match the actual dataset/propagate changes 
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
        self.AoA = self.global_params_reference["AoA"]
        self.kind = kind
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

        ## Read geometry STL file used for simulation (not same as surface mesh)
        stl_path = self.path_getter.geometry_path(car_dir)
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()

        stl_vertices = mesh_stl.points.astype(np.float32)
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ].astype(np.float32)  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten().astype(np.float32)
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"]).astype(np.float32)
        stl_centers = np.array(mesh_stl.cell_centers().points).astype(np.float32)

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
            volume_coordinates = np.float32(volume_coordinates)
            volume_fields = np.concatenate(volume_fields, axis=-1).astype(np.float32)
        
            # Non-dimensionalize by PREF and UINFTY
            volume_fields[:, 0:1] = volume_fields[:, 0:1] / PREF 
            volume_fields[:, 1:] = volume_fields[:, 1:]   / UINFTY
        else:
            volume_fields = None
            volume_coordinates = None

        ## Read surface VTP file
        if self.model_type == "surface" or self.model_type == "combined":
            surface_filepath = self.path_getter.surface_path(car_dir)
            mesh = pv.read(surface_filepath)
            mesh = mesh.point_data_to_cell_data()

            surface_fields = []
            for name in self.surface_variables:
                surface_fields.append(np.array(mesh.cell_data[name]).astype(np.float32))
            surface_fields = np.array(surface_fields)
            surface_fields = np.stack(surface_fields, axis=-1)
         
            surface_normals_area = np.array(mesh.cell_data["N_BF"]).astype(np.float32)
            surface_areas = np.linalg.norm(surface_normals_area, axis=1).astype(np.float32)
            surface_normals = np.array(mesh.cell_data["N_BF"])//np.reshape(surface_areas, (-1, 1))
            surface_coordinates = mesh.cell_centers().points.astype(np.float32)

            # Non-dimensionalize by PREF
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
        
        ## Log min/max for all returned variables
        print(
            f"stl_coordinates: shape={stl_vertices.shape}, "
            f"min={stl_vertices.min():.4f}, max={stl_vertices.max():.4f}"
        )
        print(
            f"stl_centers: shape={stl_centers.shape}, "
            f"min={stl_centers.min():.4f}, max={stl_centers.max():.4f}"
        )
        print(
            f"stl_faces: shape={mesh_indices_flattened.shape}, "
            f"min={mesh_indices_flattened.min():.4f}, "
            f"max={mesh_indices_flattened.max():.4f}"
        )
        print(
            f"stl_areas: shape={stl_sizes.shape}, "
            f"min={stl_sizes.min():.4f}, max={stl_sizes.max():.4f}"
        )

        if surface_coordinates is not None:
            print(
                f"surface_mesh_centers: shape={surface_coordinates.shape}, "
                f"min={surface_coordinates.min():.4f}, "
                f"max={surface_coordinates.max():.4f}"
            )
            print(
                f"surface_normals: shape={surface_normals.shape}, "
                f"min={surface_normals.min():.4f}, "
                f"max={surface_normals.max():.4f}"
            )
            print(
                f"surface_areas: shape={surface_areas.shape}, "
                f"min={surface_areas.min():.4f}, max={surface_areas.max():.4f}"
            )
            print(
                f"surface_fields: shape={surface_fields.shape}, "
                f"min={surface_fields.min():.4f}, "
                f"max={surface_fields.max():.4f}"
            )

        if volume_coordinates is not None:
            print(
                f"volume_mesh_centers: shape={volume_coordinates.shape}, "
                f"min={volume_coordinates.min():.4f}, "
                f"max={volume_coordinates.max():.4f}"
            )
            print(
                f"volume_fields: shape={volume_fields.shape}, "
                f"min={volume_fields.min():.4f}, max={volume_fields.max():.4f}"
            )

        print(f"global_params_values: {global_params_values}")
        print(f"global_params_reference: {global_params_reference}")

        return {
            "stl_coordinates": (stl_vertices),
            "stl_centers": (stl_centers),
            "stl_faces": (mesh_indices_flattened),
            "stl_areas": (stl_sizes),
            "surface_mesh_centers": (surface_coordinates),
            "surface_normals": (surface_normals),
            "surface_areas": (surface_areas),
            "surface_fields": (surface_fields),
            "volume_fields": (volume_fields),
            "volume_mesh_centers": (volume_coordinates),
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