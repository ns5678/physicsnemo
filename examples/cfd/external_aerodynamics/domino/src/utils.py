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

from dataclasses import dataclass
from typing import Dict, Optional, Any
import numpy as np
import torch
import pickle
from pathlib import Path
from typing import Literal


def get_num_vars(cfg: dict, model_type: Literal["volume", "surface", "combined"]):
    """Calculate the number of variables for volume, surface, and global features.

    This function analyzes the configuration to determine how many variables are needed
    for different mesh data types based on the model type. Vector variables contribute
    3 components (x, y, z) while scalar variables contribute 1 component each.

    Args:
        cfg: Configuration object containing variable definitions for volume, surface,
             and global parameters with their types (scalar/vector).
        model_type (str): Type of model - can be "volume", "surface", or "combined".
                         Determines which variable types are included in the count.

    Returns:
        tuple: A 3-tuple containing:
            - num_vol_vars (int or None): Number of volume variables. None if model_type
              is not "volume" or "combined".
            - num_surf_vars (int or None): Number of surface variables. None if model_type
              is not "surface" or "combined".
            - num_global_features (int): Number of global parameter features.
    """
    num_vol_vars = 0
    volume_variable_names = []
    if model_type == "volume" or model_type == "combined":
        volume_variable_names = list(cfg.variables.volume.solution.keys())
        for j in volume_variable_names:
            if cfg.variables.volume.solution[j] == "vector":
                num_vol_vars += 3
            else:
                num_vol_vars += 1
    else:
        num_vol_vars = None

    num_surf_vars = 0
    surface_variable_names = []
    if model_type == "surface" or model_type == "combined":
        surface_variable_names = list(cfg.variables.surface.solution.keys())
        num_surf_vars = 0
        for j in surface_variable_names:
            if cfg.variables.surface.solution[j] == "vector":
                num_surf_vars += 3
            else:
                num_surf_vars += 1
    else:
        num_surf_vars = None

    num_global_features = 0
    global_params_names = list(cfg.variables.global_parameters.keys())
    for param in global_params_names:
        if cfg.variables.global_parameters[param].type == "vector":
            num_global_features += len(cfg.variables.global_parameters[param].reference)
        elif cfg.variables.global_parameters[param].type == "scalar":
            num_global_features += 1
        else:
            raise ValueError(f"Unknown global parameter type")

    return num_vol_vars, num_surf_vars, num_global_features


@dataclass
class ScalingFactors:
    """
    Data structure for storing scaling factors computed for DoMINO datasets.

    This class provides a clean, easily serializable format for storing
    mean, std, min, and max values for different array keys in the dataset.
    Uses numpy arrays for easy serialization and cross-platform compatibility.

    Attributes:
        mean: Dictionary mapping keys to mean numpy arrays
        std: Dictionary mapping keys to standard deviation numpy arrays
        min_val: Dictionary mapping keys to minimum value numpy arrays
        max_val: Dictionary mapping keys to maximum value numpy arrays
        field_keys: List of field keys for which statistics were computed
    """

    mean: Dict[str, np.ndarray]
    std: Dict[str, np.ndarray]
    min_val: Dict[str, np.ndarray]
    max_val: Dict[str, np.ndarray]
    field_keys: list[str]

    def to_torch(
        self, device: Optional[torch.device] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Convert numpy arrays to torch tensors for use in training/inference."""
        device = device or torch.device("cpu")

        return {
            "mean": {k: torch.from_numpy(v).to(device) for k, v in self.mean.items()},
            "std": {k: torch.from_numpy(v).to(device) for k, v in self.std.items()},
            "min_val": {
                k: torch.from_numpy(v).to(device) for k, v in self.min_val.items()
            },
            "max_val": {
                k: torch.from_numpy(v).to(device) for k, v in self.max_val.items()
            },
        }

    def save(self, filepath: str | Path) -> None:
        """Save scaling factors to pickle file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str | Path) -> "ScalingFactors":
        """Load scaling factors from pickle file."""
        with open(filepath, "rb") as f:
            factors = pickle.load(f)
        return factors

    def get_field_shapes(self) -> Dict[str, tuple]:
        """Get the shape of each field's statistics."""
        return {key: self.mean[key].shape for key in self.field_keys}

    def summary(self) -> str:
        """Generate a human-readable summary of the scaling factors."""
        summary = ["Scaling Factors Summary:"]
        summary.append(f"Field Keys: {self.field_keys}")

        for key in self.field_keys:
            mean_val = self.mean[key]
            std_val = self.std[key]
            min_val = self.min_val[key]
            max_val = self.max_val[key]

            summary.append(f"\n{key}:")
            summary.append(f"  Shape: {mean_val.shape}")
            summary.append(f"  Mean: {mean_val}")
            summary.append(f"  Std: {std_val}")
            summary.append(f"  Min: {min_val}")
            summary.append(f"  Max: {max_val}")

        return "\n".join(summary)
