# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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
NVIDIA Warp implementation of FVM residual computation.

This module provides differentiable FVM residuals using NVIDIA Warp,
with support for both full mesh and batched (subset) computation.
"""

import numpy as np
import warp as wp

# Initialize Warp
wp.init()


@wp.kernel
def compute_all_residuals_warp_full(
    n_cells: int,
    points: wp.array(dtype=wp.vec3),
    velocity_data: wp.array(dtype=wp.vec3),
    pressure_data: wp.array(dtype=float),
    nut_data: wp.array(dtype=float),
    nu: float,
    cell_volumes: wp.array(dtype=float),
    cell_centers: wp.array(dtype=wp.vec3),
    cell_point_ids_flat: wp.array(dtype=int),
    cell_point_ids_offsets: wp.array(dtype=int),
    neighbors_flat: wp.array(dtype=int),
    neighbors_offsets: wp.array(dtype=int),
    face_point_ids_flat: wp.array(dtype=int),
    face_offsets: wp.array(dtype=int),
    continuity: wp.array(dtype=float),
    momentum_x: wp.array(dtype=float),
    momentum_y: wp.array(dtype=float),
    momentum_z: wp.array(dtype=float)
):
    """
    Compute FVM residuals for all cells using Warp (full mesh version).
    
    This is the Warp equivalent of the CUDA kernel in compure_physics_loss_standalone.py
    """
    idx = wp.tid()
    
    if idx >= n_cells:
        return
    
    # Get cell center
    cell1_center = cell_centers[idx]
    
    # Get cell point ids for this cell
    start_pt = cell_point_ids_offsets[idx]
    end_pt = cell_point_ids_offsets[idx + 1]
    n_pts = end_pt - start_pt
    
    if n_pts == 0:
        continuity[idx] = 0.0
        momentum_x[idx] = 0.0
        momentum_y[idx] = 0.0
        momentum_z[idx] = 0.0
        return
    
    # Compute cell-averaged velocity, pressure, nut
    cell_velocity = wp.vec3(0.0, 0.0, 0.0)
    cell_pressure = float(0.0)  # Explicit type for Warp dynamic loops
    cell_nut = float(0.0)  # Explicit type for Warp dynamic loops
    
    for pt_idx in range(n_pts):
        pt_id = cell_point_ids_flat[start_pt + pt_idx]
        cell_velocity = cell_velocity + velocity_data[pt_id]
        cell_pressure = cell_pressure + pressure_data[pt_id]
        cell_nut = cell_nut + nut_data[pt_id]
    
    cell_velocity = cell_velocity / float(n_pts)
    cell_pressure = cell_pressure / float(n_pts)
    cell_nut = cell_nut / float(n_pts)
    nu_eff = nu + cell_nut
    
    cell_volume = cell_volumes[idx]
    
    # Get neighbors for this cell
    start_nb = neighbors_offsets[idx]
    end_nb = neighbors_offsets[idx + 1]
    n_neighbors = end_nb - start_nb
    
    if n_neighbors == 0 or n_neighbors > 128:
        continuity[idx] = 0.0
        momentum_x[idx] = 0.0
        momentum_y[idx] = 0.0
        momentum_z[idx] = 0.0
        return
    
    # Compute velocity gradients using Green-Gauss
    u_grad = wp.vec3(0.0, 0.0, 0.0)
    v_grad = wp.vec3(0.0, 0.0, 0.0)
    w_grad = wp.vec3(0.0, 0.0, 0.0)
    
    # First pass: compute gradients
    for nb_idx in range(n_neighbors):
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > 16:
            continue
        
        # Compute face area and normal
        p0 = points[face_point_ids_flat[face_start]]
        area_vec = wp.vec3(0.0, 0.0, 0.0)
        
        for i in range(1, n_face_pts - 1):
            p1 = points[face_point_ids_flat[face_start + i]]
            p2 = points[face_point_ids_flat[face_start + i + 1]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            area_vec = area_vec + wp.cross(edge1, edge2)
        
        area_mag = wp.length(area_vec)
        if area_mag < 1e-30:
            continue
            
        area = 0.5 * area_mag
        normal = area_vec / area_mag
        
        # Compute face center
        face_center = wp.vec3(0.0, 0.0, 0.0)
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            face_center = face_center + points[pt_id]
        face_center = face_center / float(n_face_pts)
        
        # Check normal direction
        if wp.dot(normal, cell1_center - face_center) > 0.0:
            normal = -normal
        
        # Compute face scalar values (distance-weighted interpolation)
        u_face = float(0.0)  # Explicit type for Warp
        v_face = float(0.0)  # Explicit type for Warp
        w_face = float(0.0)  # Explicit type for Warp
        weight_sum = float(0.0)  # Explicit type for Warp
        
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            dist_sq = wp.length_sq(points[pt_id] - face_center)
            
            if dist_sq < 1e-30:
                u_face = velocity_data[pt_id][0]
                v_face = velocity_data[pt_id][1]
                w_face = velocity_data[pt_id][2]
                weight_sum = 1.0
                break
            
            weight = 1.0 / wp.sqrt(dist_sq)
            u_face = u_face + weight * velocity_data[pt_id][0]
            v_face = v_face + weight * velocity_data[pt_id][1]
            w_face = w_face + weight * velocity_data[pt_id][2]
            weight_sum = weight_sum + weight
        
        if weight_sum > 0.0:
            u_face = u_face / weight_sum
            v_face = v_face / weight_sum
            w_face = w_face / weight_sum
        
        # Accumulate gradients
        u_grad = u_grad + area * u_face * normal
        v_grad = v_grad + area * v_face * normal
        w_grad = w_grad + area * w_face * normal
    
    # Normalize gradients by cell volume
    u_grad = u_grad / cell_volume
    v_grad = v_grad / cell_volume
    w_grad = w_grad / cell_volume
    
    # Compute viscous stress tensor
    # tau_ij = nu_eff * (du_i/dx_j + du_j/dx_i)
    tau_00 = nu_eff * (u_grad[0] + u_grad[0])
    tau_01 = nu_eff * (u_grad[1] + v_grad[0])
    tau_02 = nu_eff * (u_grad[2] + w_grad[0])
    tau_10 = nu_eff * (v_grad[0] + u_grad[1])
    tau_11 = nu_eff * (v_grad[1] + v_grad[1])
    tau_12 = nu_eff * (v_grad[2] + w_grad[1])
    tau_20 = nu_eff * (w_grad[0] + u_grad[2])
    tau_21 = nu_eff * (w_grad[1] + v_grad[2])
    tau_22 = nu_eff * (w_grad[2] + w_grad[2])
    
    # Second pass: compute fluxes
    continuity_cell = float(0.0)  # Explicit type for Warp
    momentum_cell = wp.vec3(0.0, 0.0, 0.0)
    
    for nb_idx in range(n_neighbors):
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > 16:
            continue
        
        # Recompute face area and normal
        p0 = points[face_point_ids_flat[face_start]]
        area_vec = wp.vec3(0.0, 0.0, 0.0)
        
        for i in range(1, n_face_pts - 1):
            p1 = points[face_point_ids_flat[face_start + i]]
            p2 = points[face_point_ids_flat[face_start + i + 1]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            area_vec = area_vec + wp.cross(edge1, edge2)
        
        area_mag = wp.length(area_vec)
        if area_mag < 1e-30:
            continue
            
        area = 0.5 * area_mag
        normal = area_vec / area_mag
        
        face_center = wp.vec3(0.0, 0.0, 0.0)
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            face_center = face_center + points[pt_id]
        face_center = face_center / float(n_face_pts)
        
        if wp.dot(normal, cell1_center - face_center) > 0.0:
            normal = -normal
        
        # Compute face velocity and pressure
        vel_face = wp.vec3(0.0, 0.0, 0.0)
        pressure_face = float(0.0)  # Explicit type for Warp
        weight_sum = float(0.0)  # Explicit type for Warp
        
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            dist_sq = wp.length_sq(points[pt_id] - face_center)
            
            if dist_sq < 1e-30:
                vel_face = velocity_data[pt_id]
                pressure_face = pressure_data[pt_id]
                weight_sum = 1.0
                break
            
            weight = 1.0 / wp.sqrt(dist_sq)
            vel_face = vel_face + weight * velocity_data[pt_id]
            pressure_face = pressure_face + weight * pressure_data[pt_id]
            weight_sum = weight_sum + weight
        
        if weight_sum > 0.0:
            vel_face = vel_face / weight_sum
            pressure_face = pressure_face / weight_sum
        
        # Continuity flux
        continuity_cell = continuity_cell + area * wp.dot(normal, vel_face)
        
        # Momentum fluxes
        # Convective flux: area * outer(vel_face, vel_face) @ normal
        conv_flux = wp.vec3(
            wp.dot(vel_face, normal) * vel_face[0],
            wp.dot(vel_face, normal) * vel_face[1],
            wp.dot(vel_face, normal) * vel_face[2]
        )
        momentum_cell = momentum_cell + area * conv_flux
        
        # Pressure flux: area * pressure_face * normal
        momentum_cell = momentum_cell + area * pressure_face * normal
        
        # Viscous flux: area * tau @ normal
        visc_flux = wp.vec3(
            tau_00 * normal[0] + tau_01 * normal[1] + tau_02 * normal[2],
            tau_10 * normal[0] + tau_11 * normal[1] + tau_12 * normal[2],
            tau_20 * normal[0] + tau_21 * normal[1] + tau_22 * normal[2]
        )
        momentum_cell = momentum_cell - area * visc_flux
    
    continuity[idx] = continuity_cell
    momentum_x[idx] = momentum_cell[0]
    momentum_y[idx] = momentum_cell[1]
    momentum_z[idx] = momentum_cell[2]


@wp.kernel
def compute_all_residuals_warp_batch(
    cell_indices: wp.array(dtype=int),
    n_cells_compute: int,
    points: wp.array(dtype=wp.vec3),
    velocity_data: wp.array(dtype=wp.vec3),
    pressure_data: wp.array(dtype=float),
    nut_data: wp.array(dtype=float),
    nu: float,
    cell_volumes: wp.array(dtype=float),
    cell_centers: wp.array(dtype=wp.vec3),
    cell_point_ids_flat: wp.array(dtype=int),
    cell_point_ids_offsets: wp.array(dtype=int),
    neighbors_flat: wp.array(dtype=int),
    neighbors_offsets: wp.array(dtype=int),
    face_point_ids_flat: wp.array(dtype=int),
    face_offsets: wp.array(dtype=int),
    continuity: wp.array(dtype=float),
    momentum_x: wp.array(dtype=float),
    momentum_y: wp.array(dtype=float),
    momentum_z: wp.array(dtype=float)
):
    """
    Compute FVM residuals for a batch (subset) of cells using Warp.
    
    Similar to full version but operates on a subset specified by cell_indices.
    """
    tid = wp.tid()
    
    if tid >= n_cells_compute:
        return
    
    idx = cell_indices[tid]
    
    # Get cell center
    cell1_center = cell_centers[idx]
    
    # Get cell point ids for this cell
    start_pt = cell_point_ids_offsets[idx]
    end_pt = cell_point_ids_offsets[idx + 1]
    n_pts = end_pt - start_pt
    
    if n_pts == 0:
        continuity[tid] = 0.0
        momentum_x[tid] = 0.0
        momentum_y[tid] = 0.0
        momentum_z[tid] = 0.0
        return
    
    # Compute cell-averaged velocity, pressure, nut
    cell_velocity = wp.vec3(0.0, 0.0, 0.0)
    cell_pressure = float(0.0)  # Explicit type for Warp dynamic loops
    cell_nut = float(0.0)  # Explicit type for Warp dynamic loops
    
    for pt_idx in range(n_pts):
        pt_id = cell_point_ids_flat[start_pt + pt_idx]
        cell_velocity = cell_velocity + velocity_data[pt_id]
        cell_pressure = cell_pressure + pressure_data[pt_id]
        cell_nut = cell_nut + nut_data[pt_id]
    
    cell_velocity = cell_velocity / float(n_pts)
    cell_pressure = cell_pressure / float(n_pts)
    cell_nut = cell_nut / float(n_pts)
    nu_eff = nu + cell_nut
    
    cell_volume = cell_volumes[idx]
    
    # Get neighbors for this cell
    start_nb = neighbors_offsets[idx]
    end_nb = neighbors_offsets[idx + 1]
    n_neighbors = end_nb - start_nb
    
    if n_neighbors == 0 or n_neighbors > 128:
        continuity[tid] = 0.0
        momentum_x[tid] = 0.0
        momentum_y[tid] = 0.0
        momentum_z[tid] = 0.0
        return
    
    # Compute velocity gradients using Green-Gauss
    u_grad = wp.vec3(0.0, 0.0, 0.0)
    v_grad = wp.vec3(0.0, 0.0, 0.0)
    w_grad = wp.vec3(0.0, 0.0, 0.0)
    
    # First pass: compute gradients
    for nb_idx in range(n_neighbors):
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > 16:
            continue
        
        # Compute face area and normal
        p0 = points[face_point_ids_flat[face_start]]
        area_vec = wp.vec3(0.0, 0.0, 0.0)
        
        for i in range(1, n_face_pts - 1):
            p1 = points[face_point_ids_flat[face_start + i]]
            p2 = points[face_point_ids_flat[face_start + i + 1]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            area_vec = area_vec + wp.cross(edge1, edge2)
        
        area_mag = wp.length(area_vec)
        if area_mag < 1e-30:
            continue
            
        area = 0.5 * area_mag
        normal = area_vec / area_mag
        
        # Compute face center
        face_center = wp.vec3(0.0, 0.0, 0.0)
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            face_center = face_center + points[pt_id]
        face_center = face_center / float(n_face_pts)
        
        # Check normal direction
        if wp.dot(normal, cell1_center - face_center) > 0.0:
            normal = -normal
        
        # Compute face scalar values (distance-weighted interpolation)
        u_face = float(0.0)  # Explicit type for Warp
        v_face = float(0.0)  # Explicit type for Warp
        w_face = float(0.0)  # Explicit type for Warp
        weight_sum = float(0.0)  # Explicit type for Warp
        
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            dist_sq = wp.length_sq(points[pt_id] - face_center)
            
            if dist_sq < 1e-30:
                u_face = velocity_data[pt_id][0]
                v_face = velocity_data[pt_id][1]
                w_face = velocity_data[pt_id][2]
                weight_sum = 1.0
                break
            
            weight = 1.0 / wp.sqrt(dist_sq)
            u_face = u_face + weight * velocity_data[pt_id][0]
            v_face = v_face + weight * velocity_data[pt_id][1]
            w_face = w_face + weight * velocity_data[pt_id][2]
            weight_sum = weight_sum + weight
        
        if weight_sum > 0.0:
            u_face = u_face / weight_sum
            v_face = v_face / weight_sum
            w_face = w_face / weight_sum
        
        # Accumulate gradients
        u_grad = u_grad + area * u_face * normal
        v_grad = v_grad + area * v_face * normal
        w_grad = w_grad + area * w_face * normal
    
    # Normalize gradients by cell volume
    u_grad = u_grad / cell_volume
    v_grad = v_grad / cell_volume
    w_grad = w_grad / cell_volume
    
    # Compute viscous stress tensor
    tau_00 = nu_eff * (u_grad[0] + u_grad[0])
    tau_01 = nu_eff * (u_grad[1] + v_grad[0])
    tau_02 = nu_eff * (u_grad[2] + w_grad[0])
    tau_10 = nu_eff * (v_grad[0] + u_grad[1])
    tau_11 = nu_eff * (v_grad[1] + v_grad[1])
    tau_12 = nu_eff * (v_grad[2] + w_grad[1])
    tau_20 = nu_eff * (w_grad[0] + u_grad[2])
    tau_21 = nu_eff * (w_grad[1] + v_grad[2])
    tau_22 = nu_eff * (w_grad[2] + w_grad[2])
    
    # Second pass: compute fluxes
    continuity_cell = float(0.0)  # Explicit type for Warp
    momentum_cell = wp.vec3(0.0, 0.0, 0.0)
    
    for nb_idx in range(n_neighbors):
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > 16:
            continue
        
        # Recompute face area and normal
        p0 = points[face_point_ids_flat[face_start]]
        area_vec = wp.vec3(0.0, 0.0, 0.0)
        
        for i in range(1, n_face_pts - 1):
            p1 = points[face_point_ids_flat[face_start + i]]
            p2 = points[face_point_ids_flat[face_start + i + 1]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            area_vec = area_vec + wp.cross(edge1, edge2)
        
        area_mag = wp.length(area_vec)
        if area_mag < 1e-30:
            continue
            
        area = 0.5 * area_mag
        normal = area_vec / area_mag
        
        face_center = wp.vec3(0.0, 0.0, 0.0)
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            face_center = face_center + points[pt_id]
        face_center = face_center / float(n_face_pts)
        
        if wp.dot(normal, cell1_center - face_center) > 0.0:
            normal = -normal
        
        # Compute face velocity and pressure
        vel_face = wp.vec3(0.0, 0.0, 0.0)
        pressure_face = float(0.0)  # Explicit type for Warp
        weight_sum = float(0.0)  # Explicit type for Warp
        
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            dist_sq = wp.length_sq(points[pt_id] - face_center)
            
            if dist_sq < 1e-30:
                vel_face = velocity_data[pt_id]
                pressure_face = pressure_data[pt_id]
                weight_sum = 1.0
                break
            
            weight = 1.0 / wp.sqrt(dist_sq)
            vel_face = vel_face + weight * velocity_data[pt_id]
            pressure_face = pressure_face + weight * pressure_data[pt_id]
            weight_sum = weight_sum + weight
        
        if weight_sum > 0.0:
            vel_face = vel_face / weight_sum
            pressure_face = pressure_face / weight_sum
        
        # Continuity flux
        continuity_cell = continuity_cell + area * wp.dot(normal, vel_face)
        
        # Momentum fluxes
        conv_flux = wp.vec3(
            wp.dot(vel_face, normal) * vel_face[0],
            wp.dot(vel_face, normal) * vel_face[1],
            wp.dot(vel_face, normal) * vel_face[2]
        )
        momentum_cell = momentum_cell + area * conv_flux
        
        # Pressure flux
        momentum_cell = momentum_cell + area * pressure_face * normal
        
        # Viscous flux
        visc_flux = wp.vec3(
            tau_00 * normal[0] + tau_01 * normal[1] + tau_02 * normal[2],
            tau_10 * normal[0] + tau_11 * normal[1] + tau_12 * normal[2],
            tau_20 * normal[0] + tau_21 * normal[1] + tau_22 * normal[2]
        )
        momentum_cell = momentum_cell - area * visc_flux
    
    continuity[tid] = continuity_cell
    momentum_x[tid] = momentum_cell[0]
    momentum_y[tid] = momentum_cell[1]
    momentum_z[tid] = momentum_cell[2]


def compute_residuals_warp_full(mesh_data, nu):
    """
    Compute residuals for full mesh using Warp.
    
    Args:
        mesh_data: Dictionary with all mesh arrays
        nu: Kinematic viscosity
    
    Returns:
        tuple: (continuity, momentum_x, momentum_y, momentum_z) as numpy arrays
    """
    n_cells = len(mesh_data['cell_volumes'])
    
    # Convert to Warp arrays
    points_wp = wp.array(mesh_data['points'], dtype=wp.vec3)
    velocity_wp = wp.array(mesh_data['velocity_data'], dtype=wp.vec3)
    pressure_wp = wp.array(mesh_data['pressure_data'], dtype=float)
    nut_wp = wp.array(mesh_data['nut_data'], dtype=float)
    cell_volumes_wp = wp.array(mesh_data['cell_volumes'], dtype=float)
    cell_centers_wp = wp.array(mesh_data['cell_centers'], dtype=wp.vec3)
    cell_point_ids_flat_wp = wp.array(mesh_data['cell_point_ids_flat'], dtype=int)
    cell_point_ids_offsets_wp = wp.array(mesh_data['cell_point_ids_offsets'], dtype=int)
    neighbors_flat_wp = wp.array(mesh_data['neighbors_flat'], dtype=int)
    neighbors_offsets_wp = wp.array(mesh_data['neighbors_offsets'], dtype=int)
    face_point_ids_flat_wp = wp.array(mesh_data['face_point_ids_flat'], dtype=int)
    face_offsets_wp = wp.array(mesh_data['face_offsets'], dtype=int)
    
    # Allocate outputs
    continuity_wp = wp.zeros(n_cells, dtype=float)
    momentum_x_wp = wp.zeros(n_cells, dtype=float)
    momentum_y_wp = wp.zeros(n_cells, dtype=float)
    momentum_z_wp = wp.zeros(n_cells, dtype=float)
    
    # Launch kernel
    wp.launch(
        kernel=compute_all_residuals_warp_full,
        dim=n_cells,
        inputs=[
            n_cells,
            points_wp,
            velocity_wp,
            pressure_wp,
            nut_wp,
            nu,
            cell_volumes_wp,
            cell_centers_wp,
            cell_point_ids_flat_wp,
            cell_point_ids_offsets_wp,
            neighbors_flat_wp,
            neighbors_offsets_wp,
            face_point_ids_flat_wp,
            face_offsets_wp,
            continuity_wp,
            momentum_x_wp,
            momentum_y_wp,
            momentum_z_wp
        ]
    )
    
    # Synchronize and convert back to numpy
    wp.synchronize()
    
    return (
        continuity_wp.numpy(),
        momentum_x_wp.numpy(),
        momentum_y_wp.numpy(),
        momentum_z_wp.numpy()
    )


def compute_residuals_warp_batch(mesh_data, cell_indices, nu):
    """
    Compute residuals for a batch (subset) of cells using Warp.
    
    Args:
        mesh_data: Dictionary with all mesh arrays
        cell_indices: Array of cell indices to compute (e.g., [100, 5234, ...])
        nu: Kinematic viscosity
    
    Returns:
        tuple: (continuity, momentum_x, momentum_y, momentum_z) as numpy arrays
              with length = len(cell_indices)
    """
    n_cells_compute = len(cell_indices)
    
    # Convert to Warp arrays
    cell_indices_wp = wp.array(cell_indices, dtype=int)
    points_wp = wp.array(mesh_data['points'], dtype=wp.vec3)
    velocity_wp = wp.array(mesh_data['velocity_data'], dtype=wp.vec3)
    pressure_wp = wp.array(mesh_data['pressure_data'], dtype=float)
    nut_wp = wp.array(mesh_data['nut_data'], dtype=float)
    cell_volumes_wp = wp.array(mesh_data['cell_volumes'], dtype=float)
    cell_centers_wp = wp.array(mesh_data['cell_centers'], dtype=wp.vec3)
    cell_point_ids_flat_wp = wp.array(mesh_data['cell_point_ids_flat'], dtype=int)
    cell_point_ids_offsets_wp = wp.array(mesh_data['cell_point_ids_offsets'], dtype=int)
    neighbors_flat_wp = wp.array(mesh_data['neighbors_flat'], dtype=int)
    neighbors_offsets_wp = wp.array(mesh_data['neighbors_offsets'], dtype=int)
    face_point_ids_flat_wp = wp.array(mesh_data['face_point_ids_flat'], dtype=int)
    face_offsets_wp = wp.array(mesh_data['face_offsets'], dtype=int)
    
    # Allocate outputs (only for batch size)
    continuity_wp = wp.zeros(n_cells_compute, dtype=float)
    momentum_x_wp = wp.zeros(n_cells_compute, dtype=float)
    momentum_y_wp = wp.zeros(n_cells_compute, dtype=float)
    momentum_z_wp = wp.zeros(n_cells_compute, dtype=float)
    
    # Launch kernel
    wp.launch(
        kernel=compute_all_residuals_warp_batch,
        dim=n_cells_compute,
        inputs=[
            cell_indices_wp,
            n_cells_compute,
            points_wp,
            velocity_wp,
            pressure_wp,
            nut_wp,
            nu,
            cell_volumes_wp,
            cell_centers_wp,
            cell_point_ids_flat_wp,
            cell_point_ids_offsets_wp,
            neighbors_flat_wp,
            neighbors_offsets_wp,
            face_point_ids_flat_wp,
            face_offsets_wp,
            continuity_wp,
            momentum_x_wp,
            momentum_y_wp,
            momentum_z_wp
        ]
    )
    
    # Synchronize and convert back to numpy
    wp.synchronize()
    
    return (
        continuity_wp.numpy(),
        momentum_x_wp.numpy(),
        momentum_y_wp.numpy(),
        momentum_z_wp.numpy()
    )


def compute_residuals_warp_full_batched(mesh_data, nu, batch_size=8192):
    """
    Compute residuals for full mesh using batching (processes mesh in chunks).
    
    This demonstrates how to process a large mesh in batches, which is useful
    for memory-constrained scenarios or when you want to process selected regions.
    
    Args:
        mesh_data: Dictionary with all mesh arrays
        nu: Kinematic viscosity
        batch_size: Number of cells to process per batch
    
    Returns:
        tuple: (continuity, momentum_x, momentum_y, momentum_z) as numpy arrays
    """
    n_cells = len(mesh_data['cell_volumes'])
    
    # Allocate full output arrays
    continuity_full = np.zeros(n_cells)
    momentum_x_full = np.zeros(n_cells)
    momentum_y_full = np.zeros(n_cells)
    momentum_z_full = np.zeros(n_cells)
    
    # Process in batches
    for start_idx in range(0, n_cells, batch_size):
        end_idx = min(start_idx + batch_size, n_cells)
        cell_indices = np.arange(start_idx, end_idx, dtype=np.int32)
        
        # Compute batch
        cont_batch, mom_x_batch, mom_y_batch, mom_z_batch = compute_residuals_warp_batch(
            mesh_data, cell_indices, nu
        )
        
        # Store results
        continuity_full[start_idx:end_idx] = cont_batch
        momentum_x_full[start_idx:end_idx] = mom_x_batch
        momentum_y_full[start_idx:end_idx] = mom_y_batch
        momentum_z_full[start_idx:end_idx] = mom_z_batch
    
    return continuity_full, momentum_x_full, momentum_y_full, momentum_z_full


def extract_batched_mesh_data(mesh_data, cell_indices):
    """
    Extract mesh data arrays for a batch of cells.
    
    This prepares standalone arrays that contain only the data needed for
    the specified cells, allowing residual computation without the full mesh.
    
    Note: We keep ALL point data (points, velocity, pressure, nut) since 
    cells can reference any point. We only extract cell-specific arrays.
    
    Args:
        mesh_data: Full mesh data dictionary
        cell_indices: Array of cell indices to extract [n_batch]
    
    Returns:
        dict: Batched mesh data containing:
            - points, velocity_data, pressure_data, nut_data: Full arrays (referenced by indices)
            - cell_volumes, cell_centers: Extracted for batch [n_batch]
            - cell_point_ids_flat, cell_point_ids_offsets: Extracted and renumbered
            - neighbors_flat, neighbors_offsets: Extracted and renumbered
            - face_point_ids_flat, face_offsets: Extracted for relevant faces
    """
    cell_indices = np.asarray(cell_indices, dtype=np.int32)
    n_batch = len(cell_indices)
    
    # Keep full point data (cells can reference any point)
    batched_data = {
        'points': mesh_data['points'],
        'velocity_data': mesh_data['velocity_data'],
        'pressure_data': mesh_data['pressure_data'],
        'nut_data': mesh_data['nut_data'],
    }
    
    # Extract cell-specific arrays
    batched_data['cell_volumes'] = mesh_data['cell_volumes'][cell_indices]
    batched_data['cell_centers'] = mesh_data['cell_centers'][cell_indices]
    
    # Extract cell point IDs
    cell_point_ids_list = []
    for i, cell_idx in enumerate(cell_indices):
        start = mesh_data['cell_point_ids_offsets'][cell_idx]
        end = mesh_data['cell_point_ids_offsets'][cell_idx + 1]
        cell_point_ids_list.append(mesh_data['cell_point_ids_flat'][start:end])
    
    batched_data['cell_point_ids_flat'] = np.concatenate(cell_point_ids_list) if cell_point_ids_list else np.array([], dtype=np.int32)
    batched_data['cell_point_ids_offsets'] = np.concatenate([
        [0], np.cumsum([len(x) for x in cell_point_ids_list])
    ]).astype(np.int32)
    
    # Extract neighbors and faces
    neighbors_list = []
    face_point_ids_list = []
    face_offsets_list = [[0]]  # Start with 0
    
    for i, cell_idx in enumerate(cell_indices):
        nb_start = mesh_data['neighbors_offsets'][cell_idx]
        nb_end = mesh_data['neighbors_offsets'][cell_idx + 1]
        n_neighbors = nb_end - nb_start
        
        neighbors_list.append(mesh_data['neighbors_flat'][nb_start:nb_end])
        
        # Extract faces for these neighbors
        for nb_idx in range(n_neighbors):
            face_start = mesh_data['face_offsets'][nb_start + nb_idx]
            face_end = mesh_data['face_offsets'][nb_start + nb_idx + 1]
            face_points = mesh_data['face_point_ids_flat'][face_start:face_end]
            face_point_ids_list.append(face_points)
            face_offsets_list.append([len(face_points)])
    
    batched_data['neighbors_flat'] = np.concatenate(neighbors_list) if neighbors_list else np.array([], dtype=np.int32)
    batched_data['neighbors_offsets'] = np.concatenate([
        [0], np.cumsum([len(x) for x in neighbors_list])
    ]).astype(np.int32)
    
    batched_data['face_point_ids_flat'] = np.concatenate(face_point_ids_list) if face_point_ids_list else np.array([], dtype=np.int32)
    batched_data['face_offsets'] = np.concatenate([[0], np.cumsum([x[0] for x in face_offsets_list[1:]])]).astype(np.int32)
    
    return batched_data


def compute_residuals_warp_prebatched(batched_mesh_data, nu):
    """
    Compute residuals on pre-batched mesh data.
    
    This function works with mesh data that has already been extracted for
    a subset of cells using extract_batched_mesh_data(). It's more efficient
    when you need to compute residuals multiple times on the same batch.
    
    Args:
        batched_mesh_data: Dictionary with batched mesh arrays from extract_batched_mesh_data()
        nu: Kinematic viscosity
    
    Returns:
        tuple: (continuity, momentum_x, momentum_y, momentum_z) as numpy arrays
              with length = number of cells in batch
    """
    n_cells = len(batched_mesh_data['cell_volumes'])
    
    # Convert to Warp arrays
    points_wp = wp.array(batched_mesh_data['points'], dtype=wp.vec3)
    velocity_wp = wp.array(batched_mesh_data['velocity_data'], dtype=wp.vec3)
    pressure_wp = wp.array(batched_mesh_data['pressure_data'], dtype=float)
    nut_wp = wp.array(batched_mesh_data['nut_data'], dtype=float)
    cell_volumes_wp = wp.array(batched_mesh_data['cell_volumes'], dtype=float)
    cell_centers_wp = wp.array(batched_mesh_data['cell_centers'], dtype=wp.vec3)
    cell_point_ids_flat_wp = wp.array(batched_mesh_data['cell_point_ids_flat'], dtype=int)
    cell_point_ids_offsets_wp = wp.array(batched_mesh_data['cell_point_ids_offsets'], dtype=int)
    neighbors_flat_wp = wp.array(batched_mesh_data['neighbors_flat'], dtype=int)
    neighbors_offsets_wp = wp.array(batched_mesh_data['neighbors_offsets'], dtype=int)
    face_point_ids_flat_wp = wp.array(batched_mesh_data['face_point_ids_flat'], dtype=int)
    face_offsets_wp = wp.array(batched_mesh_data['face_offsets'], dtype=int)
    
    # Allocate outputs
    continuity_wp = wp.zeros(n_cells, dtype=float)
    momentum_x_wp = wp.zeros(n_cells, dtype=float)
    momentum_y_wp = wp.zeros(n_cells, dtype=float)
    momentum_z_wp = wp.zeros(n_cells, dtype=float)
    
    # Launch kernel (use full kernel since batched data is already contiguous)
    wp.launch(
        kernel=compute_all_residuals_warp_full,
        dim=n_cells,
        inputs=[
            n_cells,
            points_wp,
            velocity_wp,
            pressure_wp,
            nut_wp,
            nu,
            cell_volumes_wp,
            cell_centers_wp,
            cell_point_ids_flat_wp,
            cell_point_ids_offsets_wp,
            neighbors_flat_wp,
            neighbors_offsets_wp,
            face_point_ids_flat_wp,
            face_offsets_wp,
            continuity_wp,
            momentum_x_wp,
            momentum_y_wp,
            momentum_z_wp
        ]
    )
    
    # Synchronize and convert back to numpy
    wp.synchronize()
    
    return (
        continuity_wp.numpy(),
        momentum_x_wp.numpy(),
        momentum_y_wp.numpy(),
        momentum_z_wp.numpy()
    )


@wp.kernel
def compute_all_residuals_warp_cell_centered(
    n_cells: int,
    velocity_cell_data: wp.array(dtype=wp.vec3),
    pressure_cell_data: wp.array(dtype=float),
    nut_cell_data: wp.array(dtype=float),
    nu: float,
    cell_volumes: wp.array(dtype=float),
    cell_centers: wp.array(dtype=wp.vec3),
    neighbors_flat: wp.array(dtype=int),
    neighbors_offsets: wp.array(dtype=int),
    face_point_ids_flat: wp.array(dtype=int),
    face_offsets: wp.array(dtype=int),
    points: wp.array(dtype=wp.vec3),
    continuity: wp.array(dtype=float),
    momentum_x: wp.array(dtype=float),
    momentum_y: wp.array(dtype=float),
    momentum_z: wp.array(dtype=float)
):
    """
    Cell-centered FVM: Uses cell-averaged data directly, interpolates between cells for faces.
    
    This kernel is designed for neural network outputs which are typically cell-centered.
    No point averaging is needed - values are used directly from cells.
    Face values are interpolated between neighboring cells (distance-weighted).
    """
    idx = wp.tid()
    
    if idx >= n_cells:
        return
    
    # Get cell properties (already cell-centered, no averaging needed!)
    cell1_center = cell_centers[idx]
    cell1_velocity = velocity_cell_data[idx]
    cell1_pressure = pressure_cell_data[idx]
    cell1_nut = nut_cell_data[idx]
    nu_eff = nu + cell1_nut
    cell_volume = cell_volumes[idx]
    
    # Get neighbors
    start_nb = neighbors_offsets[idx]
    end_nb = neighbors_offsets[idx + 1]
    n_neighbors = end_nb - start_nb
    
    if n_neighbors == 0 or n_neighbors > 128:
        continuity[idx] = 0.0
        momentum_x[idx] = 0.0
        momentum_y[idx] = 0.0
        momentum_z[idx] = 0.0
        return
    
    # Compute velocity gradients using Green-Gauss
    u_grad = wp.vec3(0.0, 0.0, 0.0)
    v_grad = wp.vec3(0.0, 0.0, 0.0)
    w_grad = wp.vec3(0.0, 0.0, 0.0)
    
    # First pass: compute gradients
    for nb_idx in range(n_neighbors):
        neighbor_id = neighbors_flat[start_nb + nb_idx]
        
        # Get face geometry
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > 16:
            continue
        
        # Compute face area and normal from points
        p0 = points[face_point_ids_flat[face_start]]
        area_vec = wp.vec3(0.0, 0.0, 0.0)
        
        for i in range(1, n_face_pts - 1):
            p1 = points[face_point_ids_flat[face_start + i]]
            p2 = points[face_point_ids_flat[face_start + i + 1]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            area_vec = area_vec + wp.cross(edge1, edge2)
        
        area_mag = wp.length(area_vec)
        if area_mag < 1e-30:
            continue
            
        area = 0.5 * area_mag
        normal = area_vec / area_mag
        
        # Compute face center
        face_center = wp.vec3(0.0, 0.0, 0.0)
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            face_center = face_center + points[pt_id]
        face_center = face_center / float(n_face_pts)
        
        # Orient normal outward from cell1
        if wp.dot(normal, cell1_center - face_center) > 0.0:
            normal = -normal
        
        # Cell-to-cell interpolation for face values
        if neighbor_id >= 0 and neighbor_id < n_cells:
            # Interior face: distance-weighted interpolation
            cell2_center = cell_centers[neighbor_id]
            cell2_velocity = velocity_cell_data[neighbor_id]
            
            dist1 = wp.length(face_center - cell1_center)
            dist2 = wp.length(face_center - cell2_center)
            total_dist = dist1 + dist2
            
            if total_dist > 1e-30:
                w1 = dist2 / total_dist  # Inverse distance weight
                w2 = dist1 / total_dist
                vel_face = w1 * cell1_velocity + w2 * cell2_velocity
            else:
                vel_face = 0.5 * (cell1_velocity + cell2_velocity)
            
            u_face = vel_face[0]
            v_face = vel_face[1]
            w_face = vel_face[2]
        else:
            # Boundary face: use cell1 value
            u_face = cell1_velocity[0]
            v_face = cell1_velocity[1]
            w_face = cell1_velocity[2]
        
        # Accumulate gradients
        u_grad = u_grad + area * u_face * normal
        v_grad = v_grad + area * v_face * normal
        w_grad = w_grad + area * w_face * normal
    
    # Normalize gradients
    u_grad = u_grad / cell_volume
    v_grad = v_grad / cell_volume
    w_grad = w_grad / cell_volume
    
    # Compute viscous stress tensor
    tau_00 = nu_eff * (u_grad[0] + u_grad[0])
    tau_01 = nu_eff * (u_grad[1] + v_grad[0])
    tau_02 = nu_eff * (u_grad[2] + w_grad[0])
    tau_10 = nu_eff * (v_grad[0] + u_grad[1])
    tau_11 = nu_eff * (v_grad[1] + v_grad[1])
    tau_12 = nu_eff * (v_grad[2] + w_grad[1])
    tau_20 = nu_eff * (w_grad[0] + u_grad[2])
    tau_21 = nu_eff * (w_grad[1] + v_grad[2])
    tau_22 = nu_eff * (w_grad[2] + w_grad[2])
    
    # Second pass: compute fluxes
    continuity_cell = float(0.0)
    momentum_cell = wp.vec3(0.0, 0.0, 0.0)
    
    for nb_idx in range(n_neighbors):
        neighbor_id = neighbors_flat[start_nb + nb_idx]
        
        # Get face geometry (recompute)
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > 16:
            continue
        
        p0 = points[face_point_ids_flat[face_start]]
        area_vec = wp.vec3(0.0, 0.0, 0.0)
        
        for i in range(1, n_face_pts - 1):
            p1 = points[face_point_ids_flat[face_start + i]]
            p2 = points[face_point_ids_flat[face_start + i + 1]]
            edge1 = p1 - p0
            edge2 = p2 - p0
            area_vec = area_vec + wp.cross(edge1, edge2)
        
        area_mag = wp.length(area_vec)
        if area_mag < 1e-30:
            continue
            
        area = 0.5 * area_mag
        normal = area_vec / area_mag
        
        face_center = wp.vec3(0.0, 0.0, 0.0)
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            face_center = face_center + points[pt_id]
        face_center = face_center / float(n_face_pts)
        
        if wp.dot(normal, cell1_center - face_center) > 0.0:
            normal = -normal
        
        # Cell-to-cell interpolation
        if neighbor_id >= 0 and neighbor_id < n_cells:
            cell2_center = cell_centers[neighbor_id]
            cell2_velocity = velocity_cell_data[neighbor_id]
            cell2_pressure = pressure_cell_data[neighbor_id]
            
            dist1 = wp.length(face_center - cell1_center)
            dist2 = wp.length(face_center - cell2_center)
            total_dist = dist1 + dist2
            
            if total_dist > 1e-30:
                w1 = dist2 / total_dist
                w2 = dist1 / total_dist
                vel_face = w1 * cell1_velocity + w2 * cell2_velocity
                pressure_face = w1 * cell1_pressure + w2 * cell2_pressure
            else:
                vel_face = 0.5 * (cell1_velocity + cell2_velocity)
                pressure_face = 0.5 * (cell1_pressure + cell2_pressure)
        else:
            # Boundary
            vel_face = cell1_velocity
            pressure_face = cell1_pressure
        
        # Continuity flux
        continuity_cell = continuity_cell + area * wp.dot(normal, vel_face)
        
        # Momentum fluxes
        conv_flux = wp.vec3(
            wp.dot(vel_face, normal) * vel_face[0],
            wp.dot(vel_face, normal) * vel_face[1],
            wp.dot(vel_face, normal) * vel_face[2]
        )
        momentum_cell = momentum_cell + area * conv_flux
        momentum_cell = momentum_cell + area * pressure_face * normal
        
        visc_flux = wp.vec3(
            tau_00 * normal[0] + tau_01 * normal[1] + tau_02 * normal[2],
            tau_10 * normal[0] + tau_11 * normal[1] + tau_12 * normal[2],
            tau_20 * normal[0] + tau_21 * normal[1] + tau_22 * normal[2]
        )
        momentum_cell = momentum_cell - area * visc_flux
    
    continuity[idx] = continuity_cell
    momentum_x[idx] = momentum_cell[0]
    momentum_y[idx] = momentum_cell[1]
    momentum_z[idx] = momentum_cell[2]


def compute_residuals_warp_cell_centered(mesh_data, nu):
    """
    Compute residuals using cell-centered data (no point averaging).
    
    This is the preferred method when working with neural networks that output
    cell-centered values. Face values are interpolated between neighboring cells
    rather than from point data.
    
    Args:
        mesh_data: Dictionary with mesh arrays
        nu: Kinematic viscosity
    
    Returns:
        tuple: (continuity, momentum_x, momentum_y, momentum_z) as numpy arrays
    """
    n_cells = len(mesh_data['cell_volumes'])
    
    # Use cell-centered data directly (no point averaging needed)
    # velocity_data, pressure_data, nut_data are already cell-centered
    velocity_cell = mesh_data['velocity_data']  # [n_cells, 3]
    pressure_cell = mesh_data['pressure_data']  # [n_cells]
    nut_cell = mesh_data['nut_data']  # [n_cells]
    
    # Convert to Warp arrays
    velocity_cell_wp = wp.array(velocity_cell, dtype=wp.vec3)
    pressure_cell_wp = wp.array(pressure_cell, dtype=float)
    nut_cell_wp = wp.array(nut_cell, dtype=float)
    points_wp = wp.array(mesh_data['points'], dtype=wp.vec3)
    cell_volumes_wp = wp.array(mesh_data['cell_volumes'], dtype=float)
    cell_centers_wp = wp.array(mesh_data['cell_centers'], dtype=wp.vec3)
    neighbors_flat_wp = wp.array(mesh_data['neighbors_flat'], dtype=int)
    neighbors_offsets_wp = wp.array(mesh_data['neighbors_offsets'], dtype=int)
    face_point_ids_flat_wp = wp.array(mesh_data['face_point_ids_flat'], dtype=int)
    face_offsets_wp = wp.array(mesh_data['face_offsets'], dtype=int)
    
    # Allocate outputs
    continuity_wp = wp.zeros(n_cells, dtype=float)
    momentum_x_wp = wp.zeros(n_cells, dtype=float)
    momentum_y_wp = wp.zeros(n_cells, dtype=float)
    momentum_z_wp = wp.zeros(n_cells, dtype=float)
    
    # Launch kernel
    wp.launch(
        kernel=compute_all_residuals_warp_cell_centered,
        dim=n_cells,
        inputs=[
            n_cells,
            velocity_cell_wp,
            pressure_cell_wp,
            nut_cell_wp,
            nu,
            cell_volumes_wp,
            cell_centers_wp,
            neighbors_flat_wp,
            neighbors_offsets_wp,
            face_point_ids_flat_wp,
            face_offsets_wp,
            points_wp,
            continuity_wp,
            momentum_x_wp,
            momentum_y_wp,
            momentum_z_wp
        ]
    )
    
    wp.synchronize()
    
    return (
        continuity_wp.numpy(),
        momentum_x_wp.numpy(),
        momentum_y_wp.numpy(),
        momentum_z_wp.numpy()
    )


def compute_residuals_warp_cell_centered_torch(mesh_data, nu, device='cuda:0'):
    """
    Compute residuals using cell-centered data with PyTorch autodiff support.
    
    This version maintains the computational graph for backpropagation by using
    Warp's PyTorch interop (wp.from_torch / wp.to_torch).
    
    Args:
        mesh_data: Dictionary with mesh arrays (can contain torch tensors or numpy)
        nu: Kinematic viscosity
        device: PyTorch device (default: 'cuda:0')
    
    Returns:
        tuple: (continuity, momentum_x, momentum_y, momentum_z) as torch tensors
               with gradients enabled
    """
    import torch
    
    n_cells = len(mesh_data['cell_volumes'])
    
    # Get cell-centered data
    velocity_cell = mesh_data['velocity_data']  # [n_cells, 3]
    pressure_cell = mesh_data['pressure_data']  # [n_cells]
    nut_cell = mesh_data['nut_data']  # [n_cells]
    
    # Convert to torch if needed
    if not isinstance(velocity_cell, torch.Tensor):
        velocity_cell = torch.from_numpy(velocity_cell).to(device)
    if not isinstance(pressure_cell, torch.Tensor):
        pressure_cell = torch.from_numpy(pressure_cell).to(device)
    if not isinstance(nut_cell, torch.Tensor):
        nut_cell = torch.from_numpy(nut_cell).to(device)
    
    # Ensure contiguous memory layout for Warp
    velocity_cell = velocity_cell.contiguous().float()
    pressure_cell = pressure_cell.contiguous().float()
    nut_cell = nut_cell.contiguous().float()
    
    # Convert torch tensors to Warp arrays (maintains gradient connection!)
    velocity_cell_wp = wp.from_torch(velocity_cell, dtype=wp.vec3)
    pressure_cell_wp = wp.from_torch(pressure_cell, dtype=wp.float32)
    nut_cell_wp = wp.from_torch(nut_cell, dtype=wp.float32)
    
    # Convert mesh connectivity to Warp (these don't need gradients)
    points = mesh_data['points']
    if not isinstance(points, torch.Tensor):
        points = torch.from_numpy(points).to(device)
    points_wp = wp.from_torch(points.contiguous().float(), dtype=wp.vec3)
    
    cell_volumes = mesh_data['cell_volumes']
    if not isinstance(cell_volumes, torch.Tensor):
        cell_volumes = torch.from_numpy(cell_volumes).to(device)
    cell_volumes_wp = wp.from_torch(cell_volumes.contiguous().float(), dtype=wp.float32)
    
    cell_centers = mesh_data['cell_centers']
    if not isinstance(cell_centers, torch.Tensor):
        cell_centers = torch.from_numpy(cell_centers).to(device)
    cell_centers_wp = wp.from_torch(cell_centers.contiguous().float(), dtype=wp.vec3)
    
    # Integer arrays for connectivity
    neighbors_flat_wp = wp.array(mesh_data['neighbors_flat'], dtype=int, device=device)
    neighbors_offsets_wp = wp.array(mesh_data['neighbors_offsets'], dtype=int, device=device)
    face_point_ids_flat_wp = wp.array(mesh_data['face_point_ids_flat'], dtype=int, device=device)
    face_offsets_wp = wp.array(mesh_data['face_offsets'], dtype=int, device=device)
    
    # Allocate outputs as torch tensors
    continuity_torch = torch.zeros(n_cells, dtype=torch.float32, device=device, requires_grad=True)
    momentum_x_torch = torch.zeros(n_cells, dtype=torch.float32, device=device, requires_grad=True)
    momentum_y_torch = torch.zeros(n_cells, dtype=torch.float32, device=device, requires_grad=True)
    momentum_z_torch = torch.zeros(n_cells, dtype=torch.float32, device=device, requires_grad=True)
    
    # Convert to Warp arrays
    continuity_wp = wp.from_torch(continuity_torch, dtype=wp.float32)
    momentum_x_wp = wp.from_torch(momentum_x_torch, dtype=wp.float32)
    momentum_y_wp = wp.from_torch(momentum_y_torch, dtype=wp.float32)
    momentum_z_wp = wp.from_torch(momentum_z_torch, dtype=wp.float32)
    
    # Launch kernel
    with wp.ScopedDevice(device):
        wp.launch(
            kernel=compute_all_residuals_warp_cell_centered,
            dim=n_cells,
            inputs=[
                n_cells,
                velocity_cell_wp,
                pressure_cell_wp,
                nut_cell_wp,
                nu,
                cell_volumes_wp,
                cell_centers_wp,
                neighbors_flat_wp,
                neighbors_offsets_wp,
                face_point_ids_flat_wp,
                face_offsets_wp,
                points_wp,
                continuity_wp,
                momentum_x_wp,
                momentum_y_wp,
                momentum_z_wp
            ]
        )
    
    wp.synchronize()
    
    # Return as torch tensors (gradients flow back!)
    return (
        continuity_torch,
        momentum_x_torch,
        momentum_y_torch,
        momentum_z_torch
    )
