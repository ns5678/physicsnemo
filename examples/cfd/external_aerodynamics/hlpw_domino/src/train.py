# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
DoMINO training script for the HLPW highlift dataset (surface-only).

Adapted from the DrivAerML example with simplified loss (no physics loss,
no integral loss) and pmsh backend support via config.yaml.
"""

import os
import re
import time

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tabulate import tabulate

from physicsnemo.utils.memory import unified_gpu_memory  # noqa: F401

import torchinfo
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import distribute_module

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import torch.cuda.nvtx as nvtx

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.profiling import profile, Profiler

from physicsnemo.datapipes.cae.domino_datapipe import create_domino_dataset
from physicsnemo.models.domino.model import DoMINO
from physicsnemo.models.domino.utils import create_directory

from loss import compute_loss_dict
from utils import (
    get_keys_to_read,
    get_num_vars,
    load_scaling_factors,
    compute_l2,
    all_reduce_dict,
    coordinate_distributed_environment,
)

from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

nvmlInit()


def validation_step(
    dataloader,
    model,
    device,
    logger,
    tb_writer,
    epoch_index,
    loss_fn_type=None,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    autocast_enabled=None,
):
    dm = DistributedManager()
    running_vloss = 0.0
    with torch.no_grad():
        metrics = None

        for i_batch, sampled_batched in enumerate(dataloader):
            with autocast("cuda", enabled=autocast_enabled, cache_enabled=False):
                prediction_vol, prediction_surf = model(sampled_batched)

                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    surf_loss_scaling,
                    vol_loss_scaling,
                )

            running_vloss += loss.item()
            local_metrics = compute_l2(
                prediction_surf, prediction_vol, sampled_batched, dataloader
            )
            if metrics is None:
                metrics = local_metrics
            else:
                metrics = {
                    key: metrics[key] + local_metrics[key] for key in metrics.keys()
                }

    avg_vloss = running_vloss / (i_batch + 1)
    metrics = {key: metrics[key] / (i_batch + 1) for key in metrics.keys()}

    metrics = all_reduce_dict(metrics, dm)

    if dm.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, VAL loss norm: {loss.detach().item():.5f}"
        )
        for key in metrics.keys():
            tb_writer.add_scalar(f"L2 Metrics/val/{key}", metrics[key], epoch_index)

        metrics_table = tabulate(
            [[k, v] for k, v in metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        logger.info(
            f"\nEpoch {epoch_index} VALIDATION Average Metrics:\n{metrics_table}\n"
        )

    return avg_vloss


@profile
def train_epoch(
    dataloader,
    model,
    optimizer,
    scaler,
    tb_writer,
    logger,
    gpu_handle,
    epoch_index,
    device,
    loss_fn_type,
    vol_loss_scaling=None,
    surf_loss_scaling=None,
    autocast_enabled=None,
    grad_clip_enabled=None,
    grad_max_norm=None,
):
    dm = DistributedManager()

    running_loss = 0.0
    loss_interval = 1

    gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
    start_time = time.perf_counter()
    with Profiler():
        io_start_time = time.perf_counter()
        metrics = None
        for i_batch, sampled_batched in enumerate(dataloader):
            io_end_time = time.perf_counter()

            with autocast("cuda", enabled=autocast_enabled, cache_enabled=False):
                with nvtx.range("Model Forward Pass"):
                    prediction_vol, prediction_surf = model(sampled_batched)

                loss, loss_dict = compute_loss_dict(
                    prediction_vol,
                    prediction_surf,
                    sampled_batched,
                    loss_fn_type,
                    surf_loss_scaling,
                    vol_loss_scaling,
                )

                if isinstance(prediction_vol, tuple):
                    prediction_vol = prediction_vol[0]

                local_metrics = compute_l2(
                    prediction_surf, prediction_vol, sampled_batched, dataloader
                )
                if metrics is None:
                    metrics = local_metrics
                else:
                    metrics = {
                        key: metrics[key] + local_metrics[key] for key in metrics.keys()
                    }

            loss = loss / loss_interval
            scaler.scale(loss).backward()

            if ((i_batch + 1) % loss_interval == 0) or (i_batch + 1 == len(dataloader)):
                if grad_clip_enabled:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.detach().item()
            elapsed_time = time.perf_counter() - start_time
            io_time = io_end_time - io_start_time
            start_time = time.perf_counter()
            gpu_end_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_memory_used = gpu_end_info.used / (1024**3)
            gpu_memory_delta = (gpu_end_info.used - gpu_start_info.used) / (1024**3)

            logging_string = f"Device {device}, batch processed: {i_batch + 1}\n"
            loss_string = (
                "  "
                + "\t".join(
                    [f"{key.replace('loss_', ''):<10}" for key in loss_dict.keys()]
                )
                + "\n"
            )
            loss_string += (
                "  "
                + "\t".join(
                    [f"{v.detach().item():<10.3e}" for v in loss_dict.values()]
                )
                + "\n"
            )

            logging_string += loss_string
            logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb (delta: {gpu_memory_delta:.3f})\n"
            logging_string += f"  Timings: (IO: {io_time:.2f}, Model: {elapsed_time - io_time:.2f}, Total: {elapsed_time:.2f})s\n"
            logger.info(logging_string)
            gpu_start_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            io_start_time = time.perf_counter()

    last_loss = running_loss / (i_batch + 1)
    metrics = {key: metrics[key] / (i_batch + 1) for key in metrics.keys()}
    metrics = all_reduce_dict(metrics, dm)
    if dm.rank == 0:
        logger.info(
            f" Device {device},  batch: {i_batch + 1}, loss norm: {loss.detach().item():.5f}"
        )
        tb_x = epoch_index * len(dataloader) + i_batch + 1
        tb_writer.add_scalar("Loss/train", last_loss, tb_x)
        for key in metrics.keys():
            tb_writer.add_scalar(f"L2 Metrics/train/{key}", metrics[key], epoch_index)

        metrics_table = tabulate(
            [[k, v] for k, v in metrics.items()],
            headers=["Metric", "Average Value"],
            tablefmt="pretty",
        )
        logger.info(f"\nEpoch {epoch_index} Average Metrics:\n{metrics_table}\n")

    return last_loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist_mgr = DistributedManager()

    domain_mesh, data_mesh, placements = coordinate_distributed_environment(cfg)

    if data_mesh is not None:
        data_replica_size = data_mesh.size()
        data_rank = data_mesh.get_local_rank()
    else:
        data_replica_size = dist_mgr.world_size
        data_rank = dist_mgr.rank

    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(dist_mgr.device.index)

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist_mgr)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    vol_factors, surf_factors = load_scaling_factors(cfg)

    model_type = cfg.model.model_type
    num_vol_vars, num_surf_vars, num_global_features = get_num_vars(cfg, model_type)

    keys_to_read, keys_to_read_if_available = get_keys_to_read(
        cfg, model_type, get_ground_truth=True
    )

    train_dataloader = create_domino_dataset(
        cfg,
        phase="train",
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
        device_mesh=domain_mesh,
        placements=placements,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sample_in_bbox=cfg.data.sample_in_bbox,
        sampling=cfg.data.sampling,
    )
    train_sampler = DistributedSampler(
        train_dataloader,
        num_replicas=data_replica_size,
        rank=data_rank,
        **cfg.train.sampler,
    )

    val_dataloader = create_domino_dataset(
        cfg,
        phase="val",
        keys_to_read=keys_to_read,
        keys_to_read_if_available=keys_to_read_if_available,
        vol_factors=vol_factors,
        surf_factors=surf_factors,
        device_mesh=domain_mesh,
        placements=placements,
        normalize_coordinates=cfg.data.normalize_coordinates,
        sample_in_bbox=cfg.data.sample_in_bbox,
        sampling=cfg.data.sampling,
    )
    val_sampler = DistributedSampler(
        val_dataloader,
        num_replicas=data_replica_size,
        rank=data_rank,
        **cfg.val.sampler,
    )

    model = DoMINO(
        input_features=3,
        output_features_vol=num_vol_vars,
        output_features_surf=num_surf_vars,
        global_features=num_global_features,
        model_parameters=cfg.model,
    ).to(dist_mgr.device)

    logger.info(f"Model summary:\n{torchinfo.summary(model, verbose=0, depth=2)}\n")

    if dist_mgr.world_size > 1:
        if domain_mesh is None:
            model = DistributedDataParallel(
                model,
                device_ids=[dist_mgr.local_rank],
                output_device=dist_mgr.device,
                broadcast_buffers=dist_mgr.broadcast_buffers,
                find_unused_parameters=dist_mgr.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        else:
            model = distribute_module(model, device_mesh=domain_mesh)
            model = fully_shard(model, mesh=data_mesh)

    optimizer_class = None
    if cfg.train.optimizer.name == "Adam":
        optimizer_class = torch.optim.Adam
    elif cfg.train.optimizer.name == "AdamW":
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.train.optimizer.name}")
    optimizer = optimizer_class(
        model.parameters(),
        lr=cfg.train.optimizer.lr,
        weight_decay=cfg.train.optimizer.weight_decay,
    )
    if cfg.train.lr_scheduler.name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.train.lr_scheduler.milestones,
            gamma=cfg.train.lr_scheduler.gamma,
        )
    elif cfg.train.lr_scheduler.name == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.train.lr_scheduler.T_max,
            eta_min=cfg.train.lr_scheduler.eta_min,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.train.lr_scheduler.name}")

    scaler = GradScaler()

    writer = SummaryWriter(os.path.join(cfg.output, "tensorboard"))

    epoch_number = 0

    model_save_path = os.path.join(cfg.output, "models")
    param_save_path = os.path.join(cfg.output, "param")
    best_model_path = os.path.join(model_save_path, "best_model")
    if dist_mgr.rank == 0:
        create_directory(model_save_path)
        create_directory(param_save_path)
        create_directory(best_model_path)

    if dist_mgr.world_size > 1:
        torch.distributed.barrier()

    init_epoch = load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=dist_mgr.device,
    )

    if init_epoch != 0:
        init_epoch += 1
    epoch_number = init_epoch

    numbers = []
    for filename in os.listdir(best_model_path):
        match = re.search(r"\d+\.\d*[1-9]\d*", filename)
        if match:
            numbers.append(float(match.group(0)))

    best_vloss = min(numbers) if numbers else 1_000_000.0

    for epoch in range(init_epoch, cfg.train.epochs):
        logger.info(f"Device {dist_mgr.device}, epoch {epoch_number}:")

        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        train_dataloader.dataset.set_indices(list(train_sampler))
        val_dataloader.dataset.set_indices(list(val_sampler))

        surface_scaling_loss = cfg.model.surf_loss_scaling

        model.train(True)
        epoch_start_time = time.perf_counter()
        avg_loss = train_epoch(
            dataloader=train_dataloader,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            tb_writer=writer,
            logger=logger,
            gpu_handle=gpu_handle,
            epoch_index=epoch,
            device=dist_mgr.device,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
            autocast_enabled=cfg.train.amp.enabled,
            grad_clip_enabled=cfg.train.amp.clip_grad,
            grad_max_norm=cfg.train.amp.grad_max_norm,
        )
        epoch_end_time = time.perf_counter()
        logger.info(
            f"Device {dist_mgr.device}, Epoch {epoch_number} took {epoch_end_time - epoch_start_time:.3f} seconds"
        )

        model.eval()
        avg_vloss = validation_step(
            dataloader=val_dataloader,
            model=model,
            device=dist_mgr.device,
            logger=logger,
            tb_writer=writer,
            epoch_index=epoch,
            loss_fn_type=cfg.model.loss_function,
            vol_loss_scaling=cfg.model.vol_loss_scaling,
            surf_loss_scaling=surface_scaling_loss,
            autocast_enabled=cfg.train.amp.enabled,
        )

        scheduler.step()
        logger.info(
            f"Device {dist_mgr.device} "
            f"LOSS train {avg_loss:.5f} "
            f"valid {avg_vloss:.5f} "
            f"Current lr {scheduler.get_last_lr()[0]}"
        )

        if dist_mgr.rank == 0:
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch_number,
            )
            writer.flush()

        if dist_mgr.world_size > 1:
            torch.distributed.barrier()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss

        if dist_mgr.rank == 0:
            print(f"Device {dist_mgr.device}, Best val loss {best_vloss}")

        if dist_mgr.rank == 0 and (epoch + 1) % cfg.train.checkpoint_interval == 0.0:
            save_checkpoint(
                to_absolute_path(model_save_path),
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
            )

        epoch_number += 1

        if scheduler.get_last_lr()[0] == 1e-6:
            print("Training ended")
            exit()


if __name__ == "__main__":
    main()
