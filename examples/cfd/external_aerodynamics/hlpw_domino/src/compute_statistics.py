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

"""Compute and save scaling factors for the HLPW dataset.

Uses the same CAEDataset / PmshFileReader machinery as training,
so the pmsh_config from config.yaml is respected automatically.
"""

import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.domino_datapipe import compute_scaling_factors
from utils import ScalingFactors


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("ComputeStatistics")
    logger = RankZeroLoggingWrapper(logger, dist)

    logger.info("Starting scaling factors computation for HLPW dataset")
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")

    output_dir = os.path.dirname(cfg.data.scaling_factors)
    os.makedirs(output_dir, exist_ok=True)

    if dist.world_size > 1:
        torch.distributed.barrier()

    pickle_path = output_dir + "/scaling_factors.pkl"

    try:
        scaling_factors = ScalingFactors.load(pickle_path)
        logger.info(f"Scaling factors loaded from: {pickle_path}")
    except FileNotFoundError:
        logger.info(f"Scaling factors not found at: {pickle_path}; recomputing.")
        scaling_factors = None

    if scaling_factors is None:
        logger.info("Computing scaling factors from dataset...")
        start_time = time.perf_counter()

        target_keys = ["surface_fields", "stl_centers", "surface_mesh_centers"]
        if cfg.model.model_type in ("volume", "combined"):
            target_keys.extend(["volume_fields", "volume_mesh_centers"])

        mean, std, min_val, max_val = compute_scaling_factors(
            cfg=cfg,
            input_path=cfg.data.input_dir,
            target_keys=target_keys,
            max_samples=cfg.data.max_samples_for_statistics,
        )
        mean = {k: m.cpu().numpy() for k, m in mean.items()}
        std = {k: s.cpu().numpy() for k, s in std.items()}
        min_val = {k: m.cpu().numpy() for k, m in min_val.items()}
        max_val = {k: m.cpu().numpy() for k, m in max_val.items()}

        compute_time = time.perf_counter() - start_time
        logger.info(
            f"Scaling factors computation completed in {compute_time:.2f} seconds"
        )

        scaling_factors = ScalingFactors(
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            field_keys=target_keys,
        )

        if dist.rank == 0:
            scaling_factors.save(pickle_path)
            logger.info(f"Scaling factors saved to: {pickle_path}")

            summary_path = output_dir + "/scaling_factors_summary.txt"
            with open(summary_path, "w") as f:
                f.write(scaling_factors.summary())
            logger.info(f"Summary report saved to: {summary_path}")

        logger.info(f"Field keys processed: {scaling_factors.field_keys}")
        logger.info("Scaling factors computation completed successfully!")


if __name__ == "__main__":
    main()
