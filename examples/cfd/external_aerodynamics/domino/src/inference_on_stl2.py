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
This code defines a distributed pipeline for training the DoMINO model on
CFD datasets. It includes the computation of scaling factors, instantiating
the DoMINO model and datapipe, automatically loading the most recent checkpoint,
training the model in parallel using DistributedDataParallel across multiple
GPUs, calculating the loss and updating model parameters using mixed precision.
This is a common recipe that enables training of combined models for surface and
volume as well either of them separately. Validation is also conducted every epoch,
where predictions are compared against ground truth values. The code logs training
and validation metrics to TensorBoard. The train tab in config.yaml can be used to
specify batch size, number of epochs and other training parameters.
"""

import time
import os
import re
from typing import Literal, Any

import apex
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch

DISABLE_RMM = os.environ.get("DISABLE_RMM", "False")
if not DISABLE_RMM:
    import rmm
    from rmm.allocators.torch import rmm_torch_allocator

    rmm.reinitialize(pool_allocator=True)
    torch.cuda.memory.change_current_allocator(rmm_torch_allocator)

import torchinfo
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from nvtx import annotate as nvtx_annotate
import torch.cuda.nvtx as nvtx


from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe2 import (
    DoMINODataPipe,
    create_domino_dataset,
)
from physicsnemo.datapipes.cae.drivaer_ml_dataset import (
    DrivaerMLDataset,
)

from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import sample_points_on_mesh

from utils import ScalingFactors

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time


# Initialize NVML
nvmlInit()


from physicsnemo.utils.profiling import profile, Profiler


# Profiler().enable("torch")
# Profiler().initialize()

from loss import compute_loss_dict
from utils import get_num_vars


def inference_epoch(
    dataset: DrivaerMLDataset,
    sampler: DistributedSampler,
    datapipe: DoMINODataPipe,
    model: DoMINO,
    gpu_handle: int,
    device: torch.device,
    logger: PythonLogger,
    batch_size: int = 1_024_000,
    total_points: int = 1_024_000,
) -> float:
    epoch_indices = list(sampler)

    # n_steps = total_points // batch_size
    # if n_steps * batch_size < total_points:
    #     n_steps += 1
    #     last_batch_size = total_points - n_steps * batch_size

    # Assuming here there are more than two target meshes:
    dataset.preload(epoch_indices[0])
    dataset.preload(epoch_indices[1])
    for i_batch, epoch_index in enumerate(epoch_indices):
        # Do some preloading of input data:

        data_time_start = time.perf_counter()
        if i_batch + 2 < len(epoch_indices):
            # Preload next next
            dataset.preload(epoch_indices[i_batch + 2])
        # Get the data for this index:
        sample_batched = dataset[epoch_index]
        data_time_end = time.perf_counter()
        print(f"Data {i_batch} time: {data_time_end - data_time_start:.3f} seconds")
        procesing_time_start = time.perf_counter()
        # We always need these keys, but are only reading the faces and coordinates
        # which saves on IO speed.
        # "stl_coordinates", "stl_centers", "stl_faces", "stl_areas"

        # So, do the computation of the areas and centers:
        # Center is a mean of the 3 vertices
        triangle_vertices = sample_batched["stl_coordinates"][
            sample_batched["stl_faces"].reshape((-1, 3))
        ]
        sample_batched["stl_centers"] = triangle_vertices.mean(dim=-1)
        # Area we compute from the cross product of two sides:
        d1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
        d2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
        inferred_mesh_normals = torch.linalg.cross(d1, d2, dim=1)
        normals_norm = torch.linalg.norm(inferred_mesh_normals, dim=1)
        sample_batched["stl_areas"] = 0.5 * normals_norm

        for i in range(10):
            batch_time_start = time.perf_counter()
            # Now that we have the meshes, begin to build a batch of data up for preprocessing:
            sampled_points, sampled_faces, sampled_areas, sampled_normals = (
                sample_points_on_mesh(
                    sample_batched["stl_coordinates"],
                    sample_batched["stl_faces"],
                    batch_size,
                    mesh_normals=sample_batched["surface_normals"],
                    mesh_areas=sample_batched["stl_areas"],
                )
            )

            # Build up volume points too:
            c_min = datapipe.config.bounding_box_dims[1]
            c_max = datapipe.config.bounding_box_dims[0]

            sampled_volume_points = (c_max - c_min) * torch.rand(
                batch_size, 3, device=device, dtype=torch.float32
            ) + c_min

            inference_dict = {
                "stl_coordinates": sample_batched["stl_coordinates"],
                "stl_faces": sample_batched["stl_faces"],
                "stl_centers": sample_batched["stl_centers"],
                "stl_areas": sample_batched["stl_areas"],
                "surface_mesh_centers": sampled_points,
                "surface_normals": sampled_normals,
                "surface_areas": sampled_areas,
                "surface_faces": sampled_faces,
                "volume_mesh_centers": sampled_volume_points,
            }

            preprocessed_data = datapipe.process_data(inference_dict, i_batch)

            # Add a batch dimension to the data_dict
            preprocessed_data = {
                k: v.unsqueeze(0) for k, v in preprocessed_data.items()
            }

            with torch.no_grad():
                output_data = model(preprocessed_data)

            batch_time_end = time.perf_counter()
            points_per_second = batch_size / (batch_time_end - batch_time_start)
            print(
                f"Batch {i} in {i_batch} time: {batch_time_end - batch_time_start:.3f} seconds, {points_per_second:.3f} points per second"
            )
        procesing_time_end = time.perf_counter()
        print(
            f"Processing {i_batch} time: {procesing_time_end - procesing_time_start:.3f} seconds"
        )
        if i_batch > 20:
            break
        print(sample_batched.keys())

    return 0.0


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ################################
    # initialize distributed manager
    ################################
    DistributedManager.initialize()
    dist = DistributedManager()

    ################################
    # Initialize NVML
    ################################
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    ################################
    # Initialize logger
    ################################

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    ################################
    # Get scaling factors
    ################################
    pickle_path = os.path.join(cfg.output) + "/scaling_factors/scaling_factors.pkl"

    try:
        scaling_factors = ScalingFactors.load(pickle_path)
        logger.info(f"Scaling factors loaded from: {pickle_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaling factors not found at: {pickle_path}; please run compute_statistics.py to compute them."
        )

    model_type = cfg.model.model_type

    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    if model_type == "combined" or model_type == "surface":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
    else:
        surface_variable_names = []

    if model_type == "combined" or model_type == "volume":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
    else:
        volume_variable_names = []

    vol_factors = scaling_factors.mean["volume_fields"]
    surf_factors = scaling_factors.mean["surface_fields"]
    vol_factors_tensor = torch.from_numpy(vol_factors).to(dist.device)

    bounding_box = None

    # Override the model type
    # For the inference pipeline, we adjust the tooling a little for the data.
    # We use only a bare STL dataset that will read the mesh coordinates
    # and triangle definitions.  We'll compute the centers and normals
    # on the GPU (instead of on the CPU, as pyvista would do) and
    # then we can sample from that mesh on the GPU.
    test_dataset = DrivaerMLDataset(
        data_dir=cfg.eval.test_path,
        keys_to_read=[
            "stl_coordinates",
            "stl_faces",
        ],
        output_device=dist.device,
    )

    # Volumetric data will be generated on the fly on the GPU.

    # We _won't_ iterate over the datapipe, however, we can use the
    # datapipe processing tools on the sampled surface and
    overrides = {}
    if hasattr(cfg.data, "gpu_preprocessing"):
        overrides["gpu_preprocessing"] = cfg.data.gpu_preprocessing

    if hasattr(cfg.data, "gpu_output"):
        overrides["gpu_output"] = cfg.data.gpu_output

    test_datapipe = DoMINODataPipe(
        None,
        phase="test",
        grid_resolution=cfg.model.interp_res,
        volume_variables=volume_variable_names,
        surface_variables=surface_variable_names,
        normalize_coordinates=True,
        sampling=False,
        sample_in_bbox=True,
        volume_points_sample=None,
        surface_points_sample=None,
        geom_points_sample=None,
        positional_encoding=cfg.model.positional_encoding,
        volume_factors=vol_factors,
        surface_factors=surf_factors,
        scaling_type=cfg.model.normalization,
        model_type=model_type,
        bounding_box_dims=cfg.data.bounding_box,
        bounding_box_dims_surf=cfg.data.bounding_box_surface,
        num_surface_neighbors=cfg.model.num_neighbors_surface,
        resample_surfaces=cfg.model.resampling_surface_mesh.resample,
        resampling_points=cfg.model.resampling_surface_mesh.points,
        surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        **overrides,
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.train.sampler,
    )

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(dist.device)
    # model = torch.compile(model, fullgraph=True, dynamic=True)  # TODO make this configurable

    # Print model summary (structure and parmeter count).
    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

    model_save_path = os.path.join(cfg.output, "models")
    param_save_path = os.path.join(cfg.output, "param")
    best_model_path = os.path.join(model_save_path, "best_model")

    if dist.world_size > 1:
        torch.distributed.barrier()

    load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        device=dist.device,
    )

    initial_integral_factor_orig = cfg.model.integral_loss_scaling_factor

    start_time = time.perf_counter()

    # This controls what indices to use for each epoch.
    test_sampler.set_epoch(0)

    initial_integral_factor = initial_integral_factor_orig

    model.eval()
    epoch_start_time = time.perf_counter()
    inference_epoch(
        dataset=test_dataset,
        sampler=test_sampler,
        datapipe=test_datapipe,
        model=model,
        logger=logger,
        gpu_handle=gpu_handle,
        device=dist.device,
    )
    epoch_end_time = time.perf_counter()
    logger.info(
        f"Device {dist.device}, Epoch took {epoch_end_time - epoch_start_time:.3f} seconds"
    )


if __name__ == "__main__":
    # Profiler().enable("torch")
    # Profiler().initialize()
    main()
    # Profiler().finalize()
