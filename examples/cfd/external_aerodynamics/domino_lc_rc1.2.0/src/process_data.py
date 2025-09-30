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
This code runs the data processing in parallel to load OpenFoam files, process them 
and save in the npy or zarr format for faster processing in the DoMINO datapipes. Several 
parameters such as number of processors, input and output paths, etc. can be 
configured in config.yaml in the data_processing tab.
"""

import os
import multiprocessing
import hydra
import time
from pathlib import Path
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from openfoam_datapipe import OpenFoamDataset
from physicsnemo.utils.domino.utils import *
from zarr_utils import convert_to_zarr_format, write_zarr_file


def process_files(*args_list):
    ids = args_list[0]
    processor_id = args_list[1]
    fm_data = args_list[2]
    output_dir = args_list[3]
    serialization_method = args_list[4]
    zarr_options = args_list[5] if len(args_list) > 5 else {}
    
    for j in ids:
        fname = fm_data.filenames[j]
        if len(os.listdir(os.path.join(fm_data.data_path, fname))) == 0:
            print(f"Skipping {fname} - empty.")
            continue
        outname = os.path.join(output_dir, fname)
        print("Filename:%s on processor: %d" % (outname, processor_id))
        
        # Check if file already exists based on serialization method
        if serialization_method == "zarr":
            output_path = Path(f"{outname}.zarr")
            overwrite = zarr_options.get("overwrite_existing", True)
            if output_path.exists() and not overwrite:
                print(f"Skipping {fname} - zarr store already exists.")
                continue
        else:  # numpy
            filename = f"{outname}.npy"
            if os.path.exists(filename):
                print(f"Skipping {filename} - already exists.")
                continue
        
        start_time = time.time()
        data = fm_data[j]  # Returns OpenFoamDataInMemory object
        
        # Save based on serialization method
        if serialization_method == "zarr":
            # Convert to zarr format
            chunk_size_mb = zarr_options.get("chunk_size_mb", 1.0)
            compression_level = zarr_options.get("compression_level", 5)
            compression_method = zarr_options.get("compression_method", "zstd")
            zarr_data = convert_to_zarr_format(
                data,
                chunk_size_mb=chunk_size_mb,
                compression_level=compression_level,
                compression_method=compression_method,
            )
            # Write zarr file
            write_zarr_file(zarr_data, output_path, overwrite=overwrite)
        else:  # numpy (default)
            # Convert to dictionary for backward compatibility with existing numpy format
            data_dict = {
                "stl_coordinates": data.stl_coordinates,
                "stl_centers": data.stl_centers,
                "stl_faces": data.stl_faces,
                "stl_areas": data.stl_areas,
                "surface_mesh_centers": data.surface_mesh_centers,
                "surface_normals": data.surface_normals,
                "surface_areas": data.surface_areas,
                "volume_fields": data.volume_fields,
                "volume_mesh_centers": data.volume_mesh_centers,
                "surface_fields": data.surface_fields,
                "filename": data.metadata.filename,
                "global_params_values": data.metadata.global_params_values,
                "global_params_reference": data.metadata.global_params_reference,
            }
            np.save(filename, data_dict)
        
        print("Time taken for %d = %f" % (j, time.time() - start_time))


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    phase = "train"
    volume_variable_names = list(cfg.variables.volume.solution.keys())
    num_vol_vars = 0
    for j in volume_variable_names:
        if cfg.variables.volume.solution[j] == "vector":
            num_vol_vars += 3
        else:
            num_vol_vars += 1

    surface_variable_names = list(cfg.variables.surface.solution.keys())
    num_surf_vars = 0
    for j in surface_variable_names:
        if cfg.variables.surface.solution[j] == "vector":
            num_surf_vars += 3
        else:
            num_surf_vars += 1

    # Extract global parameters names and reference values
    global_params_names = list(cfg.variables.global_parameters.keys())
    global_params_reference = {name: cfg.variables.global_parameters[name]['reference'] for name in global_params_names}
    global_params_types = {name: cfg.variables.global_parameters[name]['type'] for name in global_params_names}
    
    fm_data = OpenFoamDataset(
        cfg.data_processor.input_dir,
        kind=cfg.data_processor.kind,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        global_params_types=global_params_types,
        global_params_reference=global_params_reference,
        model_type=cfg.model.model_type,
    )
    
    output_dir = cfg.data_processor.output_dir
    create_directory(output_dir)
    n_processors = cfg.data_processor.num_processors
    
    # Get serialization method and options from config (data_processor section)
    serialization_method = cfg.data_processor.get("serialization_method", "numpy")
    zarr_options = cfg.data_processor.get("zarr_options", {})
    
    print(f"Using serialization method: {serialization_method}")
    if serialization_method == "zarr":
        print(f"Zarr options: compression_method={zarr_options.get('compression_method', 'zstd')}, "
              f"compression_level={zarr_options.get('compression_level', 5)}, "
              f"chunk_size_mb={zarr_options.get('chunk_size_mb', 1.0)}, "
              f"overwrite_existing={zarr_options.get('overwrite_existing', True)}")

    num_files = len(fm_data)
    ids = np.arange(num_files)
    num_elements = int(num_files / n_processors) + 1
    process_list = []
    ctx = multiprocessing.get_context("spawn")
    for i in range(n_processors):
        if i != n_processors - 1:
            sf = ids[i * num_elements : i * num_elements + num_elements]
        else:
            sf = ids[i * num_elements :]
        # print(sf)
        process = ctx.Process(
            target=process_files,
            args=(sf, i, fm_data, output_dir, serialization_method, zarr_options),
        )

        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()


if __name__ == "__main__":
    main()