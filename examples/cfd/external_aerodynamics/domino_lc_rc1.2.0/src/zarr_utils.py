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
Utilities for writing OpenFOAM data to Zarr format.
"""

import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
import zarr
from numcodecs import Blosc

from schemas import (
    OpenFoamDataInMemory,
    OpenFoamZarrDataInMemory,
    PreparedZarrArrayInfo,
)


def prepare_array_for_zarr(
    array: np.ndarray,
    chunk_size_mb: float = 1.0,
    compression_level: int = 5,
    compression_method: str = "zstd",
) -> PreparedZarrArrayInfo:
    """Prepare array for Zarr storage with compression and chunking.

    Args:
        array: Input numpy array
        chunk_size_mb: Target chunk size in MB (default: 1.0)
        compression_level: Compression level 1-9 (default: 5)
        compression_method: Compression algorithm (default: "zstd")

    Returns:
        PreparedZarrArrayInfo with data, chunks, and compressor
    """
    if array is None:
        return None

    # Set up compressor
    compressor = Blosc(
        cname=compression_method, clevel=compression_level, shuffle=Blosc.SHUFFLE
    )

    # Calculate chunk size based on configured size in MB
    target_chunk_size = int(chunk_size_mb * 1024 * 1024)  # Convert MB to bytes
    item_size = array.itemsize
    shape = array.shape

    if len(shape) == 1:
        chunk_size = min(shape[0], target_chunk_size // item_size)
        chunks = (chunk_size,)
    else:
        # For 2D arrays, try to keep rows together
        chunk_rows = min(shape[0], max(1, target_chunk_size // (item_size * shape[1])))
        chunks = (chunk_rows, shape[1])

    return PreparedZarrArrayInfo(
        data=np.float32(array), chunks=chunks, compressor=compressor
    )


def convert_to_zarr_format(
    data: OpenFoamDataInMemory,
    chunk_size_mb: float = 1.0,
    compression_level: int = 5,
    compression_method: str = "zstd",
) -> OpenFoamZarrDataInMemory:
    """Convert OpenFoamDataInMemory to Zarr-ready format.

    Args:
        data: OpenFOAM data in memory
        chunk_size_mb: Target chunk size in MB
        compression_level: Compression level 1-9
        compression_method: Compression algorithm (default: "zstd")

    Returns:
        OpenFoamZarrDataInMemory with prepared arrays
    """
    return OpenFoamZarrDataInMemory(
        metadata=data.metadata,
        # Geometry arrays
        stl_coordinates=prepare_array_for_zarr(
            data.stl_coordinates, chunk_size_mb, compression_level, compression_method
        ),
        stl_centers=prepare_array_for_zarr(
            data.stl_centers, chunk_size_mb, compression_level, compression_method
        ),
        stl_faces=prepare_array_for_zarr(
            data.stl_faces, chunk_size_mb, compression_level, compression_method
        ),
        stl_areas=prepare_array_for_zarr(
            data.stl_areas, chunk_size_mb, compression_level, compression_method
        ),
        # Surface arrays (optional)
        surface_mesh_centers=prepare_array_for_zarr(
            data.surface_mesh_centers, chunk_size_mb, compression_level, compression_method
        ),
        surface_normals=prepare_array_for_zarr(
            data.surface_normals, chunk_size_mb, compression_level, compression_method
        ),
        surface_areas=prepare_array_for_zarr(
            data.surface_areas, chunk_size_mb, compression_level, compression_method
        ),
        surface_fields=prepare_array_for_zarr(
            data.surface_fields, chunk_size_mb, compression_level, compression_method
        ),
        # Volume arrays (optional)
        volume_mesh_centers=prepare_array_for_zarr(
            data.volume_mesh_centers, chunk_size_mb, compression_level, compression_method
        ),
        volume_fields=prepare_array_for_zarr(
            data.volume_fields, chunk_size_mb, compression_level, compression_method
        ),
    )


def write_zarr_file(
    data: OpenFoamZarrDataInMemory, output_path: Path, overwrite: bool = True
) -> None:
    """Write OpenFOAM data to Zarr format.

    Args:
        data: Zarr-prepared OpenFOAM data
        output_path: Path to output .zarr directory
        overwrite: Whether to overwrite existing files
    """
    # Check if store exists
    if output_path.exists():
        if not overwrite:
            print(f"Skipping {output_path} - already exists")
            return
        shutil.rmtree(output_path)

    # Create store
    zarr_store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=zarr_store)

    # Write metadata as attributes
    metadata_dict = asdict(data.metadata)
    # Convert numpy arrays in metadata to lists for JSON serialization
    for key, value in metadata_dict.items():
        if isinstance(value, np.ndarray):
            metadata_dict[key] = value.tolist()
    root.attrs.update(metadata_dict)

    # Write required arrays
    for field in ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]:
        array_info = getattr(data, field)
        if array_info is not None:
            root.create_dataset(
                field,
                data=array_info.data,
                chunks=array_info.chunks,
                compressor=array_info.compressor,
            )

    # Write optional arrays if present
    for field in [
        "surface_mesh_centers",
        "surface_normals",
        "surface_areas",
        "surface_fields",
        "volume_mesh_centers",
        "volume_fields",
    ]:
        array_info = getattr(data, field)
        if array_info is not None:
            root.create_dataset(
                field,
                data=array_info.data,
                chunks=array_info.chunks,
                compressor=array_info.compressor,
            )

