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
Data schemas for OpenFOAM dataset processing.
"""

from dataclasses import dataclass
from typing import Optional

import numcodecs
import numpy as np


@dataclass
class OpenFoamMetadata:
    """Metadata for OpenFOAM simulation data."""

    # Simulation identifiers
    filename: str

    # Physical parameters (global parameters from config)
    global_params_values: np.ndarray
    global_params_reference: np.ndarray


@dataclass(frozen=True)
class PreparedZarrArrayInfo:
    """Information for preparing an array for Zarr storage."""

    data: np.ndarray
    chunks: tuple[int, ...]
    compressor: numcodecs.abc.Codec


@dataclass
class OpenFoamDataInMemory:
    """Container for OpenFOAM simulation data in memory."""

    # Metadata
    metadata: OpenFoamMetadata

    # Geometry data (STL)
    stl_coordinates: np.ndarray
    stl_centers: np.ndarray
    stl_faces: np.ndarray
    stl_areas: np.ndarray

    # Surface data (optional - depends on model_type)
    surface_mesh_centers: Optional[np.ndarray] = None
    surface_normals: Optional[np.ndarray] = None
    surface_areas: Optional[np.ndarray] = None
    surface_fields: Optional[np.ndarray] = None

    # Volume data (optional - depends on model_type)
    volume_mesh_centers: Optional[np.ndarray] = None
    volume_fields: Optional[np.ndarray] = None


@dataclass
class OpenFoamZarrDataInMemory:
    """Container for OpenFOAM data prepared for Zarr storage."""

    # Metadata
    metadata: OpenFoamMetadata

    # Geometry data
    stl_coordinates: PreparedZarrArrayInfo
    stl_centers: PreparedZarrArrayInfo
    stl_faces: PreparedZarrArrayInfo
    stl_areas: PreparedZarrArrayInfo

    # Surface data
    surface_mesh_centers: Optional[PreparedZarrArrayInfo] = None
    surface_normals: Optional[PreparedZarrArrayInfo] = None
    surface_areas: Optional[PreparedZarrArrayInfo] = None
    surface_fields: Optional[PreparedZarrArrayInfo] = None

    # Volume data
    volume_mesh_centers: Optional[PreparedZarrArrayInfo] = None
    volume_fields: Optional[PreparedZarrArrayInfo] = None

