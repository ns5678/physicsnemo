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

from typing import Literal

import torch
import torch.cuda.nvtx as nvtx


def loss_fn(
    output: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["mse", "rmse"],
    padded_value: float = -10,
) -> torch.Tensor:
    """MSE or RMSE with masking for padded values."""
    mask = abs(target - padded_value) > 1e-3

    if loss_type == "rmse":
        dims = (0, 1)
    else:
        dims = None

    num = torch.sum(mask * (output - target) ** 2.0, dims)
    if loss_type == "rmse":
        denom = torch.sum(mask * (target - torch.mean(target, (0, 1))) ** 2.0, dims)
        loss = torch.mean(num / denom)
    elif loss_type == "mse":
        denom = torch.sum(mask)
        loss = torch.mean(num / denom)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Surface loss for HLPW: splits into [T(1), P(1), WSS(3)] = 5 components."""
    output_temp, output_pres, output_wss = torch.split(output, [1, 1, 3], dim=2)
    target_temp, target_pres, target_wss = torch.split(target, [1, 1, 3], dim=2)

    num_temp = torch.mean((output_temp - target_temp) ** 2.0)
    num_pres = torch.mean((output_pres - target_pres) ** 2.0)
    wss_diff_sq = torch.mean((target_wss - output_wss) ** 2.0, (0, 1))
    if loss_type == "mse":
        masked_loss_pres = num_pres
        masked_loss_temp = num_temp
        masked_loss_ws = torch.sum(wss_diff_sq)
    else:
        denom_pres = torch.mean(target_pres**2.0)
        masked_loss_pres = num_pres / denom_pres

        denom_temp = torch.mean(target_temp**2.0)
        masked_loss_temp = num_temp / denom_temp

        masked_loss_ws_num = wss_diff_sq
        masked_loss_ws_denom = torch.mean(target_wss**2.0, (0, 1))
        masked_loss_ws = torch.sum(masked_loss_ws_num / masked_loss_ws_denom)

    loss = masked_loss_pres + masked_loss_temp + masked_loss_ws

    return loss / 5.0


def compute_loss_dict(
    prediction_vol: torch.Tensor,
    prediction_surf: torch.Tensor,
    batch_inputs: dict,
    loss_fn_type: dict,
    surf_loss_scaling: float,
    vol_loss_scaling: float,
) -> tuple[torch.Tensor, dict]:
    """Compute loss terms for HLPW training (no physics loss, no integral loss)."""
    nvtx.range_push("Loss Calculation")
    total_loss_terms = []
    loss_dict = {}

    if prediction_vol is not None:
        target_vol = batch_inputs["volume_fields"]
        loss_vol = loss_fn(
            prediction_vol,
            target_vol,
            loss_fn_type.loss_type,
            padded_value=-10,
        )
        if loss_fn_type.loss_type == "mse":
            loss_vol = loss_vol * vol_loss_scaling

        loss_dict["loss_vol"] = loss_vol
        total_loss_terms.append(loss_vol)

    if prediction_surf is not None:
        target_surf = batch_inputs["surface_fields"]
        loss_surf = loss_fn_surface(
            prediction_surf,
            target_surf,
            loss_fn_type.loss_type,
        )

        if loss_fn_type.loss_type == "mse":
            loss_surf = loss_surf * surf_loss_scaling

        total_loss_terms.append(loss_surf)
        loss_dict["loss_surf"] = loss_surf

    total_loss = sum(total_loss_terms)
    loss_dict["total_loss"] = total_loss
    nvtx.range_pop()

    return total_loss, loss_dict
