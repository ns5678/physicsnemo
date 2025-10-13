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
and save in the npy format for faster processing in the DoMINO datapipes. Several
parameters such as number of processors, input and output paths, etc. can be
configured in config.yaml in the data_processing tab.
"""

import os
from pathlib import Path
from openfoam_datapipe import OpenFoamDataset, BoeingPaths
from physicsnemo.utils.domino.utils import *
import multiprocessing
import hydra, time
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf


def validate_folders(data_path, model_type):
    """
    Validate all folders in the data path sequentially before spawning parallel processes.
    Returns list of valid folder names and writes them to to_process.txt in the script directory.
    """
    if isinstance(data_path, str):
        data_path = Path(data_path)
    data_path = data_path.expanduser()

    # Write output files to the script directory (where process_data.py is located)
    script_dir = Path(__file__).parent

    print(f"\n{'=' * 60}")
    print(f"Starting validation of folders in {data_path}")
    print(f"Output files will be written to {script_dir}")
    print(f"{'=' * 60}\n")

    # Get all folders
    all_filenames = get_filenames(data_path)

    # Filter out folders named "sample" and .py/.txt files
    all_filenames = [
        fname
        for fname in all_filenames
        if fname != "sample"
        and not fname.endswith(".py")
        and not fname.endswith(".txt")
    ]

    print(f"Found {len(all_filenames)} folders to check")
    print(f"Validating folders for model_type: {model_type}\n")

    # Validate each folder
    valid_filenames = []
    skipped_filenames = []

    for idx, fname in enumerate(all_filenames, 1):
        car_dir = data_path / fname
        if car_dir.is_dir() and BoeingPaths.is_complete(car_dir, model_type):
            valid_filenames.append(fname)
            if idx % 100 == 0:
                print(f"Progress: {idx}/{len(all_filenames)} folders checked...")
        else:
            skipped_filenames.append(fname)
            print(f"Skipping incomplete folder: {fname}")

    print(f"\n{'=' * 60}")
    print(f"Validation Complete:")
    print(f"  Valid folders: {len(valid_filenames)}")
    print(f"  Skipped folders: {len(skipped_filenames)}")
    print(f"{'=' * 60}\n")

    # Write initial valid folder names to text file in script directory
    initial_output_file = script_dir / "to_process_initial.txt"
    with open(initial_output_file, "w") as f:
        for fname in valid_filenames:
            f.write(f"{fname}\n")
    print(
        f"Written {len(valid_filenames)} initially valid folder names to {initial_output_file}"
    )

    # Filter out folders that don't have N_BF flag (read from script directory)
    n_bf_exclusion_file = script_dir / "does_not_have_n_bf.txt"
    folders_without_n_bf = set()

    if n_bf_exclusion_file.exists():
        print(f"\nReading exclusion list from {n_bf_exclusion_file}")
        with open(n_bf_exclusion_file, "r") as f:
            folders_without_n_bf = set(line.strip() for line in f if line.strip())
        print(
            f"Found {len(folders_without_n_bf)} folders to exclude (missing N_BF flag)"
        )

        # Filter out folders without N_BF
        before_count = len(valid_filenames)
        valid_filenames = [
            fname for fname in valid_filenames if fname not in folders_without_n_bf
        ]
        filtered_count = before_count - len(valid_filenames)

        if filtered_count > 0:
            print(f"Filtered out {filtered_count} folders without N_BF flag")
    else:
        print(f"\nWarning: {n_bf_exclusion_file} not found, skipping N_BF filtering")

    # Write final valid folder names to text file in script directory
    final_output_file = script_dir / "to_process.txt"
    with open(final_output_file, "w") as f:
        for fname in valid_filenames:
            f.write(f"{fname}\n")
    print(
        f"Written {len(valid_filenames)} final valid folder names to {final_output_file}\n"
    )

    return valid_filenames


def process_files(*args_list):
    ids = args_list[0]
    processor_id = args_list[1]
    fm_data = args_list[2]
    output_dir = args_list[3]
    for j in ids:
        fname = fm_data.filenames[j]
        if len(os.listdir(os.path.join(fm_data.data_path, fname))) == 0:
            print(f"Skipping {fname} - empty.")
            continue
        outname = os.path.join(output_dir, fname)
        print("Filename:%s on processor: %d" % (outname, processor_id))
        filename = f"{outname}.npy"
        if os.path.exists(filename):
            print(f"Skipping {filename} - already exists.")
            continue
        start_time = time.time()
        data_dict = fm_data[j]
        np.save(filename, data_dict)
        print("Time taken for %d = %f" % (j, time.time() - start_time))


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    print(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    # Step 1: Validate folders sequentially BEFORE spawning processes
    print("\n" + "=" * 60)
    print("STEP 1: Validating folders (sequential)")
    print("=" * 60)
    validated_filenames = validate_folders(
        cfg.data_processor.input_dir, cfg.model.model_type
    )

    if len(validated_filenames) == 0:
        print("ERROR: No valid folders found. Exiting.")
        return

    # Step 2: Prepare data processing parameters
    print("=" * 60)
    print("STEP 2: Preparing data processing")
    print("=" * 60 + "\n")

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
    global_params_reference = {
        name: cfg.variables.global_parameters[name]["reference"]
        for name in global_params_names
    }
    global_params_types = {
        name: cfg.variables.global_parameters[name]["type"]
        for name in global_params_names
    }

    # Create dataset with pre-validated filenames
    fm_data = OpenFoamDataset(
        cfg.data_processor.input_dir,
        kind=cfg.data_processor.kind,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        global_params_types=global_params_types,
        global_params_reference=global_params_reference,
        model_type=cfg.model.model_type,
        validated_filenames=validated_filenames,
    )

    output_dir = cfg.data_processor.output_dir
    create_directory(output_dir)
    n_processors = cfg.data_processor.num_processors

    # Step 3: Spawn parallel processes for data processing
    print("=" * 60)
    print(
        f"STEP 3: Processing {len(fm_data)} folders with {n_processors} parallel processes"
    )
    print("=" * 60 + "\n")

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
        process = ctx.Process(target=process_files, args=(sf, i, fm_data, output_dir))

        process.start()
        process_list.append(process)

    for process in process_list:
        process.join()

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
