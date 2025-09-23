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
This code shows how to use a trained DoMINO model, with it's corresponding
preprocessing pipeline, to infer values on and around an STL mesh file.

This script uses the meshes from the DrivaerML dataset, however, the logic
is largely the same.  As an overview:
- Load the model
- Set up the preprocessor
- Loop over meshes
- In each mesh, sample random points on the surface, volume, or both
- Preprocess the points and run them through the model
- Process the STL mesh centers, too
- Collect the results and return
- Save the results to file.
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

DISABLE_RMM = os.environ.get("DISABLE_RMM", False)
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


from loss import compute_loss_dict
from utils import get_num_vars


def inference_on_single_stl(
    stl_coordinates: torch.Tensor,
    stl_faces: torch.Tensor,
    model: DoMINO,
    datapipe: DoMINODataPipe,
    batch_size: int,
    total_points: int,
    gpu_handle: int | None = None,
    logger: PythonLogger | None = None,
):
    """
    Perform model inference on a single STL mesh.

    This function will take the input mesh + faces and
    then sample the surface and volume to produce the model outputs
    at `total_points` locations in batches of `batch_size`.



    Args:
        stl_coordinates: The coordinates of the STL mesh.
        stl_faces: The faces of the STL mesh.
        model: The model to use for inference.
        datapipe: The datapipe to use for preprocessing.
        batch_size: The batch size to use for inference.
        total_points: The total number of points to process.
        gpu_handle: The GPU handle to use for inference.
        logger: The logger to use for logging.
    """
    device = stl_coordinates.device
    batch_start_time = time.perf_counter()
    ######################################################
    # The IO only reads in "stl_faces" and "stl_coordinates".
    # "stl_areas" and "stl_centers" would be computed by
    # pyvista on CPU - instead, we do it on the GPU
    # right here.
    ######################################################

    # Center is a mean of the 3 vertices
    triangle_vertices = stl_coordinates[stl_faces.reshape((-1, 3))]
    stl_centers = triangle_vertices.mean(dim=-1)
    ######################################################
    # Area we compute from the cross product of two sides:
    ######################################################
    d1 = triangle_vertices[:, 1] - triangle_vertices[:, 0]
    d2 = triangle_vertices[:, 2] - triangle_vertices[:, 0]
    stl_mesh_normals = torch.linalg.cross(d1, d2, dim=1)
    normals_norm = torch.linalg.norm(stl_mesh_normals, dim=1)
    stl_mesh_normals = stl_mesh_normals / normals_norm.unsqueeze(1)
    stl_areas = 0.5 * normals_norm

    ######################################################
    # For computing the points, we take those stl objects,
    # sample in chunks of `batch_size` until we've
    # accumulated `total_points` predictions.
    ######################################################

    batch_output_dict = {}
    N = 2
    total_points_processed = 0

    # Use these lists to build up the output tensors:
    surface_results = []
    volume_results = []

    while total_points_processed < total_points:
        inner_loop_start_time = time.perf_counter()

        ######################################################
        # Create the dictionary as the preprocessing expects:
        ######################################################
        inference_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces": stl_faces,
            "stl_centers": stl_centers,
            "stl_areas": stl_areas,
        }

        # If the surface data is part of the model, sample the surface:

        if datapipe.model_type == "surface" or datapipe.model_type == "combined":
            ######################################################
            # This function will sample points on the STL surface
            ######################################################
            sampled_points, sampled_faces, sampled_areas, sampled_normals = (
                sample_points_on_mesh(
                    stl_coordinates,
                    stl_faces,
                    batch_size,
                    mesh_normals=stl_mesh_normals,
                    mesh_areas=stl_areas,
                )
            )

            inference_dict["surface_mesh_centers"] = sampled_points
            inference_dict["surface_normals"] = sampled_normals
            inference_dict["surface_areas"] = sampled_areas
            inference_dict["surface_faces"] = sampled_faces

        # If the volume data is part of the model, sample the volume:
        if datapipe.model_type == "volume" or datapipe.model_type == "combined":
            ######################################################
            # Build up volume points too with uniform sampling
            # TODO - this doesn't filter points that are
            # internal to the mesh
            ######################################################
            c_min = datapipe.config.bounding_box_dims[1]
            c_max = datapipe.config.bounding_box_dims[0]

            sampled_volume_points = (c_max - c_min) * torch.rand(
                batch_size, 3, device=device, dtype=torch.float32
            ) + c_min

            inference_dict["volume_mesh_centers"] = (sampled_volume_points,)

        ######################################################
        # Pre-process the data with the datapipe:
        ######################################################
        preprocessed_data = datapipe.process_data(inference_dict)

        if datapipe.model_type == "volume" or datapipe.model_type == "combined":
            ######################################################
            # Use the sign of the volume SDF to filter out points
            # That are inside the STL mesh
            ######################################################
            sdf_nodes = preprocessed_data["sdf_nodes"]
            valid_volume_idx = sdf_nodes > 0
            preprocessed_data["volume_mesh_centers"] = preprocessed_data[
                "volume_mesh_centers"
            ][valid_volume_idx]

        ######################################################
        # Add a batch dimension to the data_dict
        # (normally this is added in __getitem__ of the datapipe)
        ######################################################
        preprocessed_data = {k: v.unsqueeze(0) for k, v in preprocessed_data.items()}

        ######################################################
        # Forward pass through the model:
        ######################################################
        with torch.no_grad():
            output_vol, output_surf = model(preprocessed_data)

        ######################################################
        # unnormalize the outputs with the datapipe
        # Whatever settings are configured for normalizing the
        # output fields - even though we don't have ground
        # truth here - are reused to undo that for the predictions
        ######################################################
        output_vol, output_surf = datapipe.unscale_model_outputs(
            output_vol, output_surf
        )

        surface_results.append(output_surf)
        volume_results.append(output_vol)

        total_points_processed += batch_size

        current_loop_time = time.perf_counter()

        logging_string = f"Device {device} processed {total_points_processed} points of {total_points}\n"
        if gpu_handle is not None:
            gpu_info = nvmlDeviceGetMemoryInfo(gpu_handle)
            gpu_memory_used = gpu_info.used / (1024**3)
            logging_string += f"  GPU memory used: {gpu_memory_used:.3f} Gb\n"

        logging_string += f"  Time taken since batch start: {current_loop_time - batch_start_time:.2f} seconds\n"
        logging_string += f"  iteration throughput: {batch_size / (current_loop_time - inner_loop_start_time):.1f} points per second\n"
        logging_string += f"  Batch mean throughput: {total_points_processed / (current_loop_time - batch_start_time):.1f} points per second.\n"

        if logger is not None:
            logger.info(logging_string)
        else:
            print(logging_string)

    ######################################################
    # Here at the end, get the values for the stl centers
    # by updating the previous inference dict
    # Only do this if the surface is part of the computation
    # Comments are shorter here - it's a condensed version
    # of the above logic.
    ######################################################
    if datapipe.model_type == "surface" or datapipe.model_type == "combined":
        stl_inference_dict = {
            "stl_coordinates": stl_coordinates,
            "stl_faces": stl_faces,
            "stl_centers": stl_centers,
            "stl_areas": stl_areas,
        }
        inference_dict["surface_mesh_centers"] = stl_centers
        inference_dict["surface_normals"] = stl_mesh_normals
        inference_dict["surface_areas"] = stl_areas
        inference_dict["surface_faces"] = stl_faces

        # Just reuse the previous volume samples here if needed:
        if datapipe.model_type == "combined":
            inference_dict["volume_mesh_centers"] = sampled_volume_points

        # Preprocess:
        preprocessed_data = datapipe.process_data(inference_dict)

        # Pull out the invalid volume points again, if needed:
        if datapipe.model_type == "combined":
            sdf_nodes = preprocessed_data["sdf_nodes"]
            valid_volume_idx = sdf_nodes > 0
            preprocessed_data["volume_mesh_centers"] = preprocessed_data[
                "volume_mesh_centers"
            ][valid_volume_idx]

        # Run the model forward:
        with torch.no_grad():
            preprocessed_data = {
                k: v.unsqueeze(0) for k, v in preprocessed_data.items()
            }
            _, output_surf = model(preprocessed_data)

        # Unnormalize the outputs:
        _, stl_center_results = datapipe.unscale_model_outputs(None, output_surf)

    else:
        stl_center_results = None

    # Stack up the results into one big tensor for surface and volume:
    if all([s is not None for s in surface_results]):
        surface_results = torch.cat(surface_results, dim=1)
    if all([v is not None for v in volume_results]):
        volume_results = torch.cat(volume_results, dim=0)

    return stl_center_results, surface_results, volume_results


def inference_epoch(
    dataset: DrivaerMLDataset,
    sampler: DistributedSampler,
    datapipe: DoMINODataPipe,
    model: DoMINO,
    gpu_handle: int,
    logger: PythonLogger,
    batch_size: int = 24_000,
    total_points: int = 1_024_000,
):
    ######################################################
    # Inference can run in a distributed way by coordinating
    # the indices for each rank, which the sampler does
    ######################################################

    # Convert the indices right to a list:
    epoch_indices = list(sampler)

    ######################################################
    # Assuming here there are more than two target meshes
    # This will get the IO pipe running in the background
    # While we process a dataset.
    ######################################################
    dataset.preload(epoch_indices[0])
    dataset.preload(epoch_indices[1])

    for i_batch, epoch_index in enumerate(epoch_indices):
        batch_start_time = time.perf_counter()
        ######################################################
        # Put another example in the preload queue while this
        # batch is processed
        ######################################################
        data_loading_start = time.perf_counter()
        if i_batch + 2 < len(epoch_indices):
            # Preload next next
            dataset.preload(epoch_indices[i_batch + 2])

        ######################################################
        # Get the data for this index:
        ######################################################
        sample_batched = dataset[epoch_index]
        dataloading_time = time.perf_counter() - data_loading_start

        logger.info(
            f"Batch {i_batch} data loading time: {dataloading_time:.3f} seconds"
        )

        procesing_time_start = time.perf_counter()
        stl_center_resulst, surface_results, volume_results = inference_on_single_stl(
            sample_batched["stl_coordinates"],
            sample_batched["stl_faces"],
            model,
            datapipe,
            batch_size,
            total_points,
            gpu_handle,
            logger,
        )

        ######################################################
        # Peel off pressure, velocity, nut, shear, etc.
        # Also compute drag, lift forces.
        ######################################################
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO

        procesing_time_end = time.perf_counter()
        logger.info(
            f"Batch {i_batch} GPU processing time: {procesing_time_end - procesing_time_start:.3f} seconds"
        )

        output_start_time = time.perf_counter()
        ######################################################
        # Save the outputs to file:
        ######################################################
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        # TODO
        output_end_time = time.perf_counter()
        logger.info(
            f"Batch {i_batch} output time: {output_end_time - output_start_time:.3f} seconds"
        )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    ######################################################
    # initialize distributed manager
    ######################################################
    DistributedManager.initialize()
    dist = DistributedManager()

    ######################################################
    # Initialize NVML
    ######################################################
    nvmlInit()
    gpu_handle = nvmlDeviceGetHandleByIndex(dist.device.index)

    ######################################################
    # Initialize logger
    ######################################################

    logger = PythonLogger("Train")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    ######################################################
    # Get scaling factors
    # Likely, you want to reuse the scaling factors from training.
    ######################################################
    pickle_path = os.path.join(cfg.output) + "/scaling_factors/scaling_factors.pkl"

    try:
        scaling_factors = ScalingFactors.load(pickle_path)
        logger.info(f"Scaling factors loaded from: {pickle_path}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Scaling factors not found at: {pickle_path}; please run compute_statistics.py to compute them."
        )

    vol_factors = scaling_factors.mean["volume_fields"]
    surf_factors = scaling_factors.mean["surface_fields"]

    ######################################################
    # Configure the model
    ######################################################
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

    ######################################################
    # Check that the sample size is equal.
    # unequal samples could be done but they aren't, here.s
    ######################################################
    if cfg.model.model_type == "combined":
        if cfg.model.volume_points_sample != cfg.model.surface_points_sample:
            raise ValueError(
                "Volume and surface points sample must be equal for combined model"
            )

    # Get the number of sample points:
    sample_points = (
        cfg.model.surface_points_sample
        if cfg.model.model_type == "surface"
        else cfg.model.volume_points_sample
    )

    ######################################################
    # If the batch size doesn't evenly divide
    # the num points, that's ok.  But print a warning
    # that the total points will get tweaked.
    ######################################################
    if cfg.eval.num_points % sample_points != 0:
        logger.warning(
            f"Batch size {sample_points} doesn't evenly divide num points {cfg.eval.num_points}."
        )
        logger.warning(
            f"Total points will be rounded up to {((cfg.eval.num_points // sample_points) + 1) * sample_points}."
        )

    ######################################################
    # Configure the dataset
    # We are applying preprocessing in a separate step
    # for this - so the dataset and datapipe are separate
    ######################################################

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

    ######################################################
    # Configure the datapipe
    # We _won't_ iterate over the datapipe, however, we can use the
    # datapipe processing tools on the sampled surface and
    # volume points with the same preprocessing.
    # It also is used to un-normalize the model outputs.
    ######################################################
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

    ######################################################
    # The sampler is used in multi-gpu inference to
    # coordinate the batches used for each rank.
    ######################################################
    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.world_size,
        rank=dist.rank,
        **cfg.train.sampler,
    )

    ######################################################
    # Configure the model
    # and move it to the device.
    ######################################################
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

    if dist.world_size > 1:
        torch.distributed.barrier()

    load_checkpoint(
        to_absolute_path(cfg.resume_dir),
        models=model,
        device=dist.device,
    )

    start_time = time.perf_counter()

    # This controls what indices to use for each epoch.
    test_sampler.set_epoch(0)

    prof = Profiler()

    model.eval()
    epoch_start_time = time.perf_counter()
    with prof:
        inference_epoch(
            dataset=test_dataset,
            sampler=test_sampler,
            datapipe=test_datapipe,
            model=model,
            logger=logger,
            gpu_handle=gpu_handle,
            batch_size=sample_points,
            total_points=cfg.eval.num_points,
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
