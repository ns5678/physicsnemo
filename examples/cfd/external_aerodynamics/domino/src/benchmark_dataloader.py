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
import torch
import torchinfo

from typing import Literal, Any

import apex
import numpy as np
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
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
    compute_scaling_factors,
    create_domino_dataset,
)
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.utils.domino.utils import *

# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time

# Initialize NVML
nvmlInit()


from physicsnemo.utils.profiling import profile, Profiler


@profile
def train_epoch(
    dataloader,
    sampler,
    logger,
    gpu_handle,
    epoch_index,
    device,
):
    dist = DistributedManager()

    indices = list(iter(sampler))
    print(f"indices: {indices}")
    # If you tell the dataloader the indices in advance, it will preload
    # and pre-preprocess data
    # dataloader.set_indices(indices)

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    for i_batch, sample_batched in enumerate(dataloader):
        # sampled_batched = dict_to_device(sample_batched, device)
        # if i_batch == 7:
        # break
        # for key in sampled_batched.keys():
        #     print(f"{key}: {sampled_batched[key].shape}")

        # Gather data and report
        elapsed_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
        gpu_memory_used = gpu_end_info.used / (1024**3)
        gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

        logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
        logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb\n"
        logging_string += f"  GPU memory delta: {gpu_memory_delta:.3f} Gb\n"
        logging_string += f"  Time taken: {elapsed_time:.2f} seconds\n"
        logger.info(logging_string)
        gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)

    return


def get_or_compute_scaling_factors(
    cfg: DictConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get or compute scaling factors for volume and surface fields normalization.

    This function either loads pre-computed scaling factors from disk or computes them
    if they don't exist. The scaling factors are used for normalizing volume and surface
    fields data based on the specified normalization method in the config.

    Args:
        cfg (DictConfig): Configuration object containing:
            - project.name: Project name for saving/loading scaling factors
            - model.normalization: Type of normalization ("min_max_scaling" or "mean_std_scaling")
            - data.input_dir: Input directory path
            - data_processor.use_cache: Whether to use cached data

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - vol_factors: Scaling factors for volume fields (max/min or mean/std)
            - surf_factors: Scaling factors for surface fields (max/min or mean/std)
            Each factor is a numpy array containing the respective scaling values.

    Raises:
        ValueError: If an invalid normalization type is specified in the config.
    """
    # Compute or load the scaling factors:
    vol_save_path = os.path.join(
        "outputs", cfg.project.name, "volume_scaling_factors.npy"
    )
    surf_save_path = os.path.join(
        "outputs", cfg.project.name, "surface_scaling_factors.npy"
    )

    if not os.path.exists(vol_save_path) or not os.path.exists(surf_save_path):
        # Save the scaling factors if needed:
        mean, std, min_val, max_val = compute_scaling_factors(
            cfg=cfg,
            input_path=cfg.data.input_dir,
            use_cache=cfg.data_processor.use_cache,
        )

        v_mean = mean["volume_fields"].cpu().numpy()
        v_std = std["volume_fields"].cpu().numpy()
        v_min = min_val["volume_fields"].cpu().numpy()
        v_max = max_val["volume_fields"].cpu().numpy()

        s_mean = mean["surface_fields"].cpu().numpy()
        s_std = std["surface_fields"].cpu().numpy()
        s_min = min_val["surface_fields"].cpu().numpy()
        s_max = max_val["surface_fields"].cpu().numpy()

        np.save(vol_save_path, [v_mean, v_std, v_min, v_max])
        np.save(surf_save_path, [s_mean, s_std, s_min, s_max])
    else:
        v_mean, v_std, v_min, v_max = np.load(vol_save_path)
        s_mean, s_std, s_min, s_max = np.load(surf_save_path)

    if cfg.model.normalization == "min_max_scaling":
        vol_factors = [v_max, v_min]
    elif cfg.model.normalization == "mean_std_scaling":
        vol_factors = [v_mean, v_std]
    else:
        raise ValueError(f"Invalid normalization type: {cfg.model.normalization}")

    if cfg.model.normalization == "min_max_scaling":
        surf_factors = [s_max, s_min]
    elif cfg.model.normalization == "mean_std_scaling":
        surf_factors = [s_mean, s_std]
    else:
        raise ValueError(f"Invalid normalization type: {cfg.model.normalization}")

    return vol_factors, surf_factors


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize NVML
    nvmlInit()

    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    model_type = cfg.model.model_type

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    vol_factors, surf_factors = get_or_compute_scaling_factors(cfg)

    train_dataset = create_domino_dataset(
        cfg,
        phase="train",
        volume_variable_names="volume_fields",
        surface_variable_names="surface_fields",
        vol_factors=vol_factors,
        surf_factors=surf_factors,
    )
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=dist.world_size, rank=dist.rank
    )

    # train_dataloader = DataLoader(
    #     train_dataset,
    #     sampler=train_sampler,
    #     **cfg.train.dataloader,
    # )

    for epoch in range(0, cfg.train.epochs):
        start_time = time.perf_counter()
        logger.info(f"Device {dist.device}, epoch {epoch}:")

        epoch_start_time = time.perf_counter()
        with Profiler():
            train_epoch(
                dataloader=train_dataset,
                sampler=train_sampler,
                logger=logger,
                gpu_handle=gpu_handle,
                epoch_index=epoch,
                device=dist.device,
            )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist.device}, Epoch {epoch} took {epoch_end_time - epoch_start_time:.3f} seconds"
        )


if __name__ == "__main__":
    # Profiler().enable("torch")
    # Profiler().initialize()
    main()
    # Profiler().finalize()
