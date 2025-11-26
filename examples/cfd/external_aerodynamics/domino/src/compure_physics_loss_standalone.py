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
Finite Volume Method (FVM) Residual Computation for CFD (Cell-Centered)

This module provides CPU and GPU implementations for computing continuity and 
momentum residuals using the Finite Volume Method. The implementation uses:
- Cell-centered FVM: values at cell centers, face values from cell-to-cell interpolation
- Green-Gauss gradient reconstruction
- Distance-weighted interpolation between cells at face centers
- Numba JIT compilation for CPU (parallel execution)
- CUDA kernel for GPU acceleration

Main API:
---------
compute_residuals_fvm : Compute FVM residuals on CPU or GPU

Example usage:
--------------
>>> ugrid = compute_residuals_fvm(
...     "mesh.vtu",
...     velocity_field="UMean",
...     pressure_field="pMean", 
...     nut_field="nutMean",
...     nu=1.5e-5,
...     device="gpu"
... )
"""

import os
# MUST set this before any numba imports
os.environ['NUMBA_NUM_THREADS'] = '16'

import numpy as np
from numba import njit, prange, cuda, float64
import vtk
from vtk.util import numpy_support
from tqdm import tqdm

@njit(fastmath=True)
def compute_face_area_numba(point_ids, points):
    n_points = len(point_ids)
    p0 = points[point_ids[0]]
    area_vec = np.zeros(3)
    
    for i in range(1, n_points - 1):
        p1 = points[point_ids[i]]
        p2 = points[point_ids[i + 1]]
        edge1 = p1 - p0
        edge2 = p2 - p0
        area_vec += np.cross(edge1, edge2)
    
    return 0.5 * np.sqrt(np.sum(area_vec * area_vec))

@njit(fastmath=True)
def compute_face_normal_numba(point_ids, points, cell_center):
    n_points = len(point_ids)
    p0 = points[point_ids[0]]
    area_vec = np.zeros(3)
    
    for i in range(1, n_points - 1):
        p1 = points[point_ids[i]]
        p2 = points[point_ids[i + 1]]
        edge1 = p1 - p0
        edge2 = p2 - p0
        area_vec += np.cross(edge1, edge2)
    
    normal = area_vec / (np.sqrt(np.sum(area_vec * area_vec)))
    
    face_center = np.zeros(3)
    for i in range(n_points):
        face_center += points[point_ids[i]]
    face_center /= n_points
    
    face_to_cell = cell_center - face_center
    
    if np.sum(normal * face_to_cell) > 0:
        normal = -normal
    
    return normal

@njit(fastmath=True)
def compute_face_center_numba(point_ids, points):
    """Compute face center as average of face point positions."""
    face_center = np.zeros(3)
    n = len(point_ids)
    for i in range(n):
        face_center += points[point_ids[i]]
    return face_center / n

@njit(fastmath=True)
def compute_face_velocity_numba(point_ids, velocity_data, points):
    """Distance-weighted interpolation of velocity at face center."""
    # Compute face center
    face_center = compute_face_center_numba(point_ids, points)
    
    # Distance-weighted interpolation
    vel = np.zeros(3, dtype=velocity_data.dtype)
    weight_sum = 0.0
    n = len(point_ids)
    
    for i in range(n):
        pt_id = point_ids[i]
        dist = np.sqrt(np.sum((points[pt_id] - face_center) ** 2))
        
        # Use inverse distance weighting (avoid division by zero)
        if dist < 1e-15:
            # Point is exactly at face center, use it directly
            return velocity_data[pt_id].copy()
        
        weight = 1.0 / dist
        vel += weight * velocity_data[pt_id]
        weight_sum += weight
    
    # Cast back to original dtype to maintain type consistency
    result = (vel / weight_sum).astype(velocity_data.dtype)
    return result

@njit(fastmath=True)
def compute_face_scalar_numba(point_ids, scalar_data, points):
    """Distance-weighted interpolation of scalar at face center."""
    # Compute face center
    face_center = compute_face_center_numba(point_ids, points)
    
    # Distance-weighted interpolation
    val = scalar_data.dtype.type(0.0)  # Use same dtype as scalar_data
    weight_sum = 0.0
    n = len(point_ids)
    
    for i in range(n):
        pt_id = point_ids[i]
        dist = np.sqrt(np.sum((points[pt_id] - face_center) ** 2))
        
        # Use inverse distance weighting (avoid division by zero)
        if dist < 1e-15:
            # Point is exactly at face center, use it directly
            return scalar_data[pt_id]
        
        weight = 1.0 / dist
        val += weight * scalar_data[pt_id]
        weight_sum += weight
    
    # Cast back to original dtype to maintain type consistency
    return scalar_data.dtype.type(val / weight_sum)

def extract_vtk_connectivity(mesh):
    """Extract raw connectivity arrays directly from VTK."""
    if hasattr(mesh, 'GetOutput'):
        ugrid = mesh.GetOutput()
    else:
        ugrid = mesh
    
    cells = ugrid.GetCells()
    cell_connectivity = numpy_support.vtk_to_numpy(cells.GetConnectivityArray())
    cell_offsets = numpy_support.vtk_to_numpy(cells.GetOffsetsArray())
    
    return cell_connectivity, cell_offsets, ugrid.GetNumberOfCells()

def build_face_connectivity_vtk(ugrid):
    """Build face connectivity using pure VTK API."""
    print("Building connectivity...")
    
    n_cells = ugrid.GetNumberOfCells()
    
    # Extract VTK connectivity
    cell_connectivity, cell_offsets, _ = extract_vtk_connectivity(ugrid)
    
    # Build cell point IDs
    cell_point_ids = []
    for cell_idx in range(n_cells):
        start = cell_offsets[cell_idx]
        end = cell_offsets[cell_idx + 1]
        cell_point_ids.append(list(cell_connectivity[start:end]))
    
    # Extract faces using VTK API - hash-based matching for O(1) neighbor lookup
    face_to_cells = {}
    
    for cell_idx in range(n_cells):
        if cell_idx % 100000 == 0:
            print(f"  Processing cell {cell_idx:,} / {n_cells:,}")
        
        cell = ugrid.GetCell(cell_idx)
        n_faces = cell.GetNumberOfFaces()
        
        for face_idx in range(n_faces):
            face = cell.GetFace(face_idx)
            face_point_ids_vtk = face.GetPointIds()
            n_face_pts = face_point_ids_vtk.GetNumberOfIds()
            
            # Extract face point IDs
            face_pts = [face_point_ids_vtk.GetId(i) for i in range(n_face_pts)]
            face_tuple = tuple(sorted(face_pts))
            
            if face_tuple not in face_to_cells:
                face_to_cells[face_tuple] = []
            face_to_cells[face_tuple].append((cell_idx, np.array(face_pts, dtype=np.int64)))
    
    print(f"  Extracted {len(face_to_cells):,} unique faces")
    
    # Build neighbors from face matches
    neighbors = [[] for _ in range(n_cells)]
    face_point_ids_map = {}
    
    for face_tuple, cell_list in face_to_cells.items():
        if len(cell_list) == 2:  # Internal face shared by 2 cells
            (cell1_id, face_pts1), (cell2_id, face_pts2) = cell_list
            neighbors[cell1_id].append(cell2_id)
            neighbors[cell2_id].append(cell1_id)
            face_point_ids_map[(cell1_id, cell2_id)] = face_pts1
            face_point_ids_map[(cell2_id, cell1_id)] = face_pts1
    
    print(f"  Built neighbor connectivity ({len(face_point_ids_map):,} face pairs)")
    
    return neighbors, face_point_ids_map, cell_point_ids

def compute_cell_gradient_gauss_green(cell_id, points, scalar_data, cell_volumes, cell_centers, neighbors, face_point_ids_map):
    gradient = np.zeros(3)
    cell1_center = cell_centers[cell_id]
    cell_volume = cell_volumes[cell_id]

    for neighbor_idx in neighbors[cell_id]:
        face_point_ids = face_point_ids_map[(cell_id, neighbor_idx)]
        
        area = compute_face_area_numba(face_point_ids, points)
        normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
        scalar_face = compute_face_scalar_numba(face_point_ids, scalar_data, points)
        gradient += area * scalar_face * normal
    
    return gradient / cell_volume

@njit(fastmath=True)
def compute_viscous_stress_tensor(velocity_grad, nu_eff):
    tau = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            tau[i, j] = nu_eff * (velocity_grad[i, j] + velocity_grad[j, i])
    return tau

def compute_cell_momentum(cell_id, points, velocity_data, pressure_data, nut_data, nu, cell_volumes, cell_centers, cell_point_ids, neighbors, face_point_ids_map):
    momentum_residual = np.zeros(3)
    cell1_center = cell_centers[cell_id]
    
    u_grad = compute_cell_gradient_gauss_green(cell_id, points, velocity_data[:, 0], cell_volumes, cell_centers, neighbors, face_point_ids_map)
    v_grad = compute_cell_gradient_gauss_green(cell_id, points, velocity_data[:, 1], cell_volumes, cell_centers, neighbors, face_point_ids_map)
    w_grad = compute_cell_gradient_gauss_green(cell_id, points, velocity_data[:, 2], cell_volumes, cell_centers, neighbors, face_point_ids_map)
    
    velocity_grad = np.array([u_grad, v_grad, w_grad])
    
    cell_pts = cell_point_ids[cell_id]
    cell_velocity = np.mean([velocity_data[pt_id] for pt_id in cell_pts], axis=0)
    cell_pressure = np.mean([pressure_data[pt_id] for pt_id in cell_pts])
    cell_nut = np.mean([nut_data[pt_id] for pt_id in cell_pts])
    nu_eff = nu + cell_nut
    
    tau = compute_viscous_stress_tensor(velocity_grad, nu_eff)
    
    for neighbor_idx in neighbors[cell_id]:
        face_point_ids = face_point_ids_map[(cell_id, neighbor_idx)]
        
        area = compute_face_area_numba(face_point_ids, points)
        normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
        vel_face = compute_face_velocity_numba(face_point_ids, velocity_data, points)
        pressure_face = compute_face_scalar_numba(face_point_ids, pressure_data, points)
        
        convective_flux = area * np.outer(vel_face, vel_face) @ normal
        pressure_flux = area * pressure_face * normal
        viscous_flux = area * tau @ normal
        
        momentum_residual += convective_flux + pressure_flux - viscous_flux
    
    return momentum_residual
    
def compute_cell_continuity(cell_id, points, velocity_data, cell_centers, neighbors, face_point_ids_map):
    flux = 0.0
    cell1_center = cell_centers[cell_id]

    for neighbor_idx in neighbors[cell_id]:
        if (cell_id, neighbor_idx) not in face_point_ids_map:
            continue
        
        face_point_ids = face_point_ids_map[(cell_id, neighbor_idx)]
        area = compute_face_area_numba(face_point_ids, points)
        normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
        vel_face_center = compute_face_velocity_numba(face_point_ids, velocity_data, points)
        flux += area * np.sum(normal * vel_face_center)
    
    return flux

MAX_NEIGHBORS = 128
MAX_FACE_POINTS = 16


@njit(parallel=False, fastmath=True)
def compute_all_residuals_cpu(n_cells, points, velocity_data, pressure_data, nut_data, nu, 
                              cell_volumes, cell_centers, cell_point_ids_flat, cell_point_ids_offsets,
                              neighbors_flat, neighbors_offsets, face_point_ids_flat, face_offsets):
    """
    Parallel computation of all residuals using numba prange with fastmath optimizations (CPU version).
    Cell-centered FVM: uses cell data directly, face values interpolated between cells.
    
    Arrays are flattened to work with numba's limitations on complex data structures.
    """
    continuity = np.zeros(n_cells)
    momentum_x = np.zeros(n_cells)
    momentum_y = np.zeros(n_cells)
    momentum_z = np.zeros(n_cells)
    
    for idx in prange(n_cells):
        # Get cell center
        cell1_center = cell_centers[idx]
        
        # Use cell-centered data directly (no averaging from points)
        cell_velocity = velocity_data[idx].copy()
        cell_pressure = pressure_data[idx]
        cell_nut = nut_data[idx]
        nu_eff = nu + cell_nut
        
        # Compute velocity gradients using simple Green-Gauss
        u_grad = np.zeros(3)
        v_grad = np.zeros(3)
        w_grad = np.zeros(3)
        
        cell_volume = cell_volumes[idx]
        
        # Get neighbors for this cell
        start_nb = neighbors_offsets[idx]
        end_nb = neighbors_offsets[idx + 1]
        n_neighbors = end_nb - start_nb
        
        # First pass: compute gradients
        for nb_idx in range(n_neighbors):
            neighbor_id = neighbors_flat[start_nb + nb_idx]
            
            # Get face point ids from flattened arrays
            face_start = face_offsets[start_nb + nb_idx]
            face_end = face_offsets[start_nb + nb_idx + 1]
            face_point_ids = face_point_ids_flat[face_start:face_end]
            
            if len(face_point_ids) == 0:
                continue
            
            # Compute face properties
            area = compute_face_area_numba(face_point_ids, points)
            normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
            
            # Cell-to-cell interpolation for face values
            if neighbor_id >= 0 and neighbor_id < n_cells:
                # Interior face: distance-weighted interpolation between cells
                cell2_center = cell_centers[neighbor_id]
                cell2_velocity = velocity_data[neighbor_id]
                
                # Compute face center
                face_center = np.zeros(3)
                for i in range(len(face_point_ids)):
                    face_center += points[face_point_ids[i]]
                face_center /= len(face_point_ids)
                
                dist1 = np.sqrt(np.sum((face_center - cell1_center)**2))
                dist2 = np.sqrt(np.sum((face_center - cell2_center)**2))
                total_dist = dist1 + dist2
                
                if total_dist > 1e-30:
                    w1 = dist2 / total_dist  # Inverse distance weight
                    w2 = dist1 / total_dist
                    vel_face = w1 * cell_velocity + w2 * cell2_velocity
                else:
                    vel_face = 0.5 * (cell_velocity + cell2_velocity)
                
                u_face = vel_face[0]
                v_face = vel_face[1]
                w_face = vel_face[2]
            else:
                # Boundary face: use cell1 value
                u_face = cell_velocity[0]
                v_face = cell_velocity[1]
                w_face = cell_velocity[2]
            
            u_grad += area * u_face * normal
            v_grad += area * v_face * normal
            w_grad += area * w_face * normal
        
        u_grad /= cell_volume
        v_grad /= cell_volume
        w_grad /= cell_volume
        
        velocity_grad = np.zeros((3, 3))
        velocity_grad[0, :] = u_grad
        velocity_grad[1, :] = v_grad
        velocity_grad[2, :] = w_grad
        
        # Compute viscous stress tensor
        tau = compute_viscous_stress_tensor(velocity_grad, nu_eff)
        
        # Second pass: compute fluxes
        continuity_cell = 0.0
        momentum_cell = np.zeros(3)
        
        for nb_idx in range(n_neighbors):
            neighbor_id = neighbors_flat[start_nb + nb_idx]
            
            # Get face point ids
            face_start = face_offsets[start_nb + nb_idx]
            face_end = face_offsets[start_nb + nb_idx + 1]
            face_point_ids = face_point_ids_flat[face_start:face_end]
            
            if len(face_point_ids) == 0:
                continue
            
            # Compute face properties
            area = compute_face_area_numba(face_point_ids, points)
            normal = compute_face_normal_numba(face_point_ids, points, cell1_center)
            
            # Cell-to-cell interpolation for face values
            if neighbor_id >= 0 and neighbor_id < n_cells:
                cell2_center = cell_centers[neighbor_id]
                cell2_velocity = velocity_data[neighbor_id]
                cell2_pressure = pressure_data[neighbor_id]
                
                # Compute face center
                face_center = np.zeros(3)
                for i in range(len(face_point_ids)):
                    face_center += points[face_point_ids[i]]
                face_center /= len(face_point_ids)
                
                dist1 = np.sqrt(np.sum((face_center - cell1_center)**2))
                dist2 = np.sqrt(np.sum((face_center - cell2_center)**2))
                total_dist = dist1 + dist2
                
                if total_dist > 1e-30:
                    w1 = dist2 / total_dist
                    w2 = dist1 / total_dist
                    vel_face = w1 * cell_velocity + w2 * cell2_velocity
                    pressure_face = w1 * cell_pressure + w2 * cell2_pressure
                else:
                    vel_face = 0.5 * (cell_velocity + cell2_velocity)
                    pressure_face = 0.5 * (cell_pressure + cell2_pressure)
            else:
                # Boundary
                vel_face = cell_velocity
                pressure_face = cell_pressure
            
            # Continuity flux
            continuity_cell += area * np.sum(normal * vel_face)
            
            # Momentum fluxes
            convective_flux = area * np.outer(vel_face, vel_face) @ normal
            pressure_flux = area * pressure_face * normal
            viscous_flux = area * tau @ normal
            
            momentum_cell += convective_flux + pressure_flux - viscous_flux
        
        continuity[idx] = continuity_cell
        momentum_x[idx] = momentum_cell[0]
        momentum_y[idx] = momentum_cell[1]
        momentum_z[idx] = momentum_cell[2]
    
    return continuity, momentum_x, momentum_y, momentum_z


@cuda.jit
def compute_all_residuals_gpu(n_cells, points, velocity_data, pressure_data, nut_data, nu,
                              cell_volumes, cell_centers, cell_point_ids_flat, cell_point_ids_offsets,
                              neighbors_flat, neighbors_offsets, face_point_ids_flat, face_offsets,
                              continuity, momentum_x, momentum_y, momentum_z):
    """
    GPU kernel for computing all residuals using CUDA (cell-centered FVM).
    """
    idx = cuda.grid(1)
    if idx >= n_cells:
        return
    
    # Get cell center
    cell1_center = cuda.local.array(3, dtype=float64)
    for i in range(3):
        cell1_center[i] = cell_centers[idx, i]
    
    # Use cell-centered data directly (no averaging from points)
    cell_velocity = cuda.local.array(3, dtype=float64)
    for i in range(3):
        cell_velocity[i] = velocity_data[idx, i]
    cell_pressure = pressure_data[idx]
    cell_nut = nut_data[idx]
    nu_eff = nu + cell_nut
    
    cell_volume = cell_volumes[idx]
    
    # Get neighbors for this cell
    start_nb = neighbors_offsets[idx]
    end_nb = neighbors_offsets[idx + 1]
    n_neighbors = end_nb - start_nb
    
    if n_neighbors == 0 or n_neighbors > MAX_NEIGHBORS:
        continuity[idx] = 0.0
        momentum_x[idx] = 0.0
        momentum_y[idx] = 0.0
        momentum_z[idx] = 0.0
        return
    
    # Compute velocity gradients using Green-Gauss
    u_grad = cuda.local.array(3, dtype=float64)
    v_grad = cuda.local.array(3, dtype=float64)
    w_grad = cuda.local.array(3, dtype=float64)
    for i in range(3):
        u_grad[i] = 0.0
        v_grad[i] = 0.0
        w_grad[i] = 0.0
    
    # First pass: compute gradients
    for nb_idx in range(n_neighbors):
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > MAX_FACE_POINTS:
            continue
        
        # Compute face area and normal
        p0 = cuda.local.array(3, dtype=float64)
        pt_id_0 = face_point_ids_flat[face_start]
        for i in range(3):
            p0[i] = points[pt_id_0, i]
        
        area_vec = cuda.local.array(3, dtype=float64)
        for i in range(3):
            area_vec[i] = 0.0
        
        for i in range(1, n_face_pts - 1):
            p1 = cuda.local.array(3, dtype=float64)
            p2 = cuda.local.array(3, dtype=float64)
            pt_id_1 = face_point_ids_flat[face_start + i]
            pt_id_2 = face_point_ids_flat[face_start + i + 1]
            
            for j in range(3):
                p1[j] = points[pt_id_1, j]
                p2[j] = points[pt_id_2, j]
            
            edge1 = cuda.local.array(3, dtype=float64)
            edge2 = cuda.local.array(3, dtype=float64)
            for j in range(3):
                edge1[j] = p1[j] - p0[j]
                edge2[j] = p2[j] - p0[j]
            
            # Cross product
            area_vec[0] += edge1[1] * edge2[2] - edge1[2] * edge2[1]
            area_vec[1] += edge1[2] * edge2[0] - edge1[0] * edge2[2]
            area_vec[2] += edge1[0] * edge2[1] - edge1[1] * edge2[0]
        
        area_mag = 0.0
        for i in range(3):
            area_mag += area_vec[i] * area_vec[i]
        area_mag = cuda.libdevice.sqrt(area_mag)
        area = 0.5 * area_mag
        
        # Compute normal
        normal = cuda.local.array(3, dtype=float64)
        for i in range(3):
            normal[i] = area_vec[i] / (2.0 * area_mag)
        
        # Compute face center
        face_center = cuda.local.array(3, dtype=float64)
        for i in range(3):
            face_center[i] = 0.0
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            for j in range(3):
                face_center[j] += points[pt_id, j]
        for i in range(3):
            face_center[i] /= n_face_pts
        
        # Check normal direction
        face_to_cell_dot = 0.0
        for i in range(3):
            face_to_cell_dot += normal[i] * (cell1_center[i] - face_center[i])
        
        if face_to_cell_dot > 0:
            for i in range(3):
                normal[i] = -normal[i]
        
        # Cell-to-cell interpolation for face values
        u_face = 0.0
        v_face = 0.0
        w_face = 0.0
        
        neighbor_id = neighbors_flat[start_nb + nb_idx]
        if neighbor_id >= 0 and neighbor_id < n_cells:
            # Interior face: distance-weighted interpolation between cells
            cell2_center = cuda.local.array(3, dtype=float64)
            cell2_velocity = cuda.local.array(3, dtype=float64)
            for i in range(3):
                cell2_center[i] = cell_centers[neighbor_id, i]
                cell2_velocity[i] = velocity_data[neighbor_id, i]
            
            dist1 = 0.0
            dist2 = 0.0
            for i in range(3):
                diff1 = face_center[i] - cell1_center[i]
                diff2 = face_center[i] - cell2_center[i]
                dist1 += diff1 * diff1
                dist2 += diff2 * diff2
            dist1 = cuda.libdevice.sqrt(dist1)
            dist2 = cuda.libdevice.sqrt(dist2)
            total_dist = dist1 + dist2
            
            if total_dist > 1e-30:
                w1 = dist2 / total_dist
                w2 = dist1 / total_dist
                u_face = w1 * cell_velocity[0] + w2 * cell2_velocity[0]
                v_face = w1 * cell_velocity[1] + w2 * cell2_velocity[1]
                w_face = w1 * cell_velocity[2] + w2 * cell2_velocity[2]
            else:
                u_face = 0.5 * (cell_velocity[0] + cell2_velocity[0])
                v_face = 0.5 * (cell_velocity[1] + cell2_velocity[1])
                w_face = 0.5 * (cell_velocity[2] + cell2_velocity[2])
        else:
            # Boundary face: use cell1 value
            u_face = cell_velocity[0]
            v_face = cell_velocity[1]
            w_face = cell_velocity[2]
        
        # Accumulate gradients
        for i in range(3):
            u_grad[i] += area * u_face * normal[i]
            v_grad[i] += area * v_face * normal[i]
            w_grad[i] += area * w_face * normal[i]
    
    # Normalize gradients by cell volume
    for i in range(3):
        u_grad[i] /= cell_volume
        v_grad[i] /= cell_volume
        w_grad[i] /= cell_volume
    
    # Compute viscous stress tensor
    tau = cuda.local.array((3, 3), dtype=float64)
    velocity_grad = cuda.local.array((3, 3), dtype=float64)
    velocity_grad[0, 0] = u_grad[0]
    velocity_grad[0, 1] = u_grad[1]
    velocity_grad[0, 2] = u_grad[2]
    velocity_grad[1, 0] = v_grad[0]
    velocity_grad[1, 1] = v_grad[1]
    velocity_grad[1, 2] = v_grad[2]
    velocity_grad[2, 0] = w_grad[0]
    velocity_grad[2, 1] = w_grad[1]
    velocity_grad[2, 2] = w_grad[2]
    
    for i in range(3):
        for j in range(3):
            tau[i, j] = nu_eff * (velocity_grad[i, j] + velocity_grad[j, i])
    
    # Second pass: compute fluxes
    continuity_cell = 0.0
    momentum_cell = cuda.local.array(3, dtype=float64)
    for i in range(3):
        momentum_cell[i] = 0.0
    
    for nb_idx in range(n_neighbors):
        face_start = face_offsets[start_nb + nb_idx]
        face_end = face_offsets[start_nb + nb_idx + 1]
        n_face_pts = face_end - face_start
        
        if n_face_pts == 0 or n_face_pts > MAX_FACE_POINTS:
            continue
        
        # Recompute face area and normal (same as above)
        p0 = cuda.local.array(3, dtype=float64)
        pt_id_0 = face_point_ids_flat[face_start]
        for i in range(3):
            p0[i] = points[pt_id_0, i]
        
        area_vec = cuda.local.array(3, dtype=float64)
        for i in range(3):
            area_vec[i] = 0.0
        
        for i in range(1, n_face_pts - 1):
            p1 = cuda.local.array(3, dtype=float64)
            p2 = cuda.local.array(3, dtype=float64)
            pt_id_1 = face_point_ids_flat[face_start + i]
            pt_id_2 = face_point_ids_flat[face_start + i + 1]
            
            for j in range(3):
                p1[j] = points[pt_id_1, j]
                p2[j] = points[pt_id_2, j]
            
            edge1 = cuda.local.array(3, dtype=float64)
            edge2 = cuda.local.array(3, dtype=float64)
            for j in range(3):
                edge1[j] = p1[j] - p0[j]
                edge2[j] = p2[j] - p0[j]
            
            area_vec[0] += edge1[1] * edge2[2] - edge1[2] * edge2[1]
            area_vec[1] += edge1[2] * edge2[0] - edge1[0] * edge2[2]
            area_vec[2] += edge1[0] * edge2[1] - edge1[1] * edge2[0]
        
        area_mag = 0.0
        for i in range(3):
            area_mag += area_vec[i] * area_vec[i]
        area_mag = cuda.libdevice.sqrt(area_mag)
        area = 0.5 * area_mag
        
        normal = cuda.local.array(3, dtype=float64)
        for i in range(3):
            normal[i] = area_vec[i] / area_mag
        
        face_center = cuda.local.array(3, dtype=float64)
        for i in range(3):
            face_center[i] = 0.0
        for i in range(n_face_pts):
            pt_id = face_point_ids_flat[face_start + i]
            for j in range(3):
                face_center[j] += points[pt_id, j]
        for i in range(3):
            face_center[i] /= n_face_pts
        
        face_to_cell_dot = 0.0
        for i in range(3):
            face_to_cell_dot += normal[i] * (cell1_center[i] - face_center[i])
        
        if face_to_cell_dot > 0:
            for i in range(3):
                normal[i] = -normal[i]
        
        # Cell-to-cell interpolation for face values
        vel_face = cuda.local.array(3, dtype=float64)
        pressure_face = 0.0
        
        neighbor_id = neighbors_flat[start_nb + nb_idx]
        if neighbor_id >= 0 and neighbor_id < n_cells:
            # Interior face: distance-weighted interpolation between cells
            cell2_center = cuda.local.array(3, dtype=float64)
            cell2_velocity = cuda.local.array(3, dtype=float64)
            for i in range(3):
                cell2_center[i] = cell_centers[neighbor_id, i]
                cell2_velocity[i] = velocity_data[neighbor_id, i]
            cell2_pressure = pressure_data[neighbor_id]
            
            dist1 = 0.0
            dist2 = 0.0
            for i in range(3):
                diff1 = face_center[i] - cell1_center[i]
                diff2 = face_center[i] - cell2_center[i]
                dist1 += diff1 * diff1
                dist2 += diff2 * diff2
            dist1 = cuda.libdevice.sqrt(dist1)
            dist2 = cuda.libdevice.sqrt(dist2)
            total_dist = dist1 + dist2
            
            if total_dist > 1e-30:
                w1 = dist2 / total_dist
                w2 = dist1 / total_dist
                for i in range(3):
                    vel_face[i] = w1 * cell_velocity[i] + w2 * cell2_velocity[i]
                pressure_face = w1 * cell_pressure + w2 * cell2_pressure
            else:
                for i in range(3):
                    vel_face[i] = 0.5 * (cell_velocity[i] + cell2_velocity[i])
                pressure_face = 0.5 * (cell_pressure + cell2_pressure)
        else:
            # Boundary face: use cell1 value
            for i in range(3):
                vel_face[i] = cell_velocity[i]
            pressure_face = cell_pressure
        
        # Continuity flux
        vel_normal_dot = 0.0
        for i in range(3):
            vel_normal_dot += normal[i] * vel_face[i]
        continuity_cell += area * vel_normal_dot
        
        # Momentum fluxes
        # Convective flux: area * outer(vel_face, vel_face) @ normal
        for i in range(3):
            conv_flux = 0.0
            for j in range(3):
                conv_flux += vel_face[i] * vel_face[j] * normal[j]
            momentum_cell[i] += area * conv_flux
        
        # Pressure flux: area * pressure_face * normal
        for i in range(3):
            momentum_cell[i] += area * pressure_face * normal[i]
        
        # Viscous flux: area * tau @ normal
        for i in range(3):
            visc_flux = 0.0
            for j in range(3):
                visc_flux += tau[i, j] * normal[j]
            momentum_cell[i] -= area * visc_flux
    
    continuity[idx] = continuity_cell
    momentum_x[idx] = momentum_cell[0]
    momentum_y[idx] = momentum_cell[1]
    momentum_z[idx] = momentum_cell[2]


def _prepare_mesh_data(ugrid, velocity_field, pressure_field, nut_field):
    """
    Prepare mesh data for FVM residual computation (cell-centered approach).
    
    Parameters
    ----------
    ugrid : vtk.vtkUnstructuredGrid
        The input unstructured grid.
    velocity_field : str
        Name of the velocity field in cell data.
    pressure_field : str
        Name of the pressure field in cell data.
    nut_field : str
        Name of the turbulent viscosity field in cell data.
    
    Returns
    -------
    dict
        Dictionary containing all prepared arrays and connectivity information.
    """
    # Convert point data to cell data if needed
    if velocity_field not in ugrid.GetCellData().keys():
        p2c = vtk.vtkPointDataToCellData()
        p2c.SetInputData(ugrid)
        p2c.PassPointDataOn()
        p2c.Update()
        ugrid = p2c.GetOutput()
    
    # Compute cell volumes
    if not ugrid.GetCellData().HasArray("Volume"):
        cell_size_filter = vtk.vtkCellSizeFilter()
        cell_size_filter.SetInputData(ugrid)
        cell_size_filter.SetComputeLength(False)
        cell_size_filter.SetComputeArea(False)
        cell_size_filter.SetComputeVolume(True)
        cell_size_filter.SetComputeVertexCount(False)
        cell_size_filter.Update()
        ugrid = cell_size_filter.GetOutput()
    
    # Extract data arrays (cell-centered)
    points = numpy_support.vtk_to_numpy(ugrid.GetPoints().GetData()).astype(np.float64)
    velocity_data = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray(velocity_field)).astype(np.float64)
    pressure_data = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray(pressure_field)).astype(np.float64)
    nut_data = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray(nut_field)).astype(np.float64)
    cell_volumes = numpy_support.vtk_to_numpy(ugrid.GetCellData().GetArray("Volume")).astype(np.float64)
    
    # Compute cell centers
    cell_centers_filter = vtk.vtkCellCenters()
    cell_centers_filter.SetInputData(ugrid)
    cell_centers_filter.Update()
    cell_centers = numpy_support.vtk_to_numpy(cell_centers_filter.GetOutput().GetPoints().GetData()).astype(np.float64)
    
    # Build face connectivity
    neighbors, face_point_ids_map, cell_point_ids = build_face_connectivity_vtk(ugrid)
    
    # Flatten data structures for numba parallel processing
    cell_point_ids_flat = []
    cell_point_ids_offsets = [0]
    for cell_pts in cell_point_ids:
        cell_point_ids_flat.extend(cell_pts)
        cell_point_ids_offsets.append(len(cell_point_ids_flat))
    
    neighbors_flat = []
    neighbors_offsets = [0]
    face_point_ids_flat = []
    face_offsets = [0]
    
    for cell_id, cell_neighbors in enumerate(neighbors):
        neighbors_flat.extend(cell_neighbors)
        neighbors_offsets.append(len(neighbors_flat))
        
        for neighbor_id in cell_neighbors:
            if (cell_id, neighbor_id) in face_point_ids_map:
                face_pts = face_point_ids_map[(cell_id, neighbor_id)]
                face_point_ids_flat.extend(face_pts)
            face_offsets.append(len(face_point_ids_flat))
    
    # Convert to numpy arrays
    cell_point_ids_flat = np.array(cell_point_ids_flat, dtype=np.int64)
    cell_point_ids_offsets = np.array(cell_point_ids_offsets, dtype=np.int64)
    neighbors_flat = np.array(neighbors_flat, dtype=np.int64)
    neighbors_offsets = np.array(neighbors_offsets, dtype=np.int64)
    face_point_ids_flat = np.array(face_point_ids_flat, dtype=np.int64)
    face_offsets = np.array(face_offsets, dtype=np.int64)
    
    return {
        'ugrid': ugrid,
        'points': points,
        'velocity_data': velocity_data,
        'pressure_data': pressure_data,
        'nut_data': nut_data,
        'cell_volumes': cell_volumes,
        'cell_centers': cell_centers,
        'cell_point_ids_flat': cell_point_ids_flat,
        'cell_point_ids_offsets': cell_point_ids_offsets,
        'neighbors_flat': neighbors_flat,
        'neighbors_offsets': neighbors_offsets,
        'face_point_ids_flat': face_point_ids_flat,
        'face_offsets': face_offsets,
    }


def _compute_residuals_fvm(mesh_data, nu, device="cpu"):
    """
    Internal function to compute FVM residuals on CPU or GPU.
    
    Parameters
    ----------
    mesh_data : dict
        Dictionary containing prepared mesh data and connectivity.
    nu : float
        Kinematic viscosity.
    device : str
        Device to use: "cpu" or "gpu".
    
    Returns
    -------
    tuple
        (continuity, momentum_x, momentum_y, momentum_z) residuals as numpy arrays.
    """
    n_cells = len(mesh_data['cell_volumes'])
    
    if device == "cpu":
        continuity, momentum_x, momentum_y, momentum_z = compute_all_residuals_cpu(
            n_cells,
            mesh_data['points'],
            mesh_data['velocity_data'],
            mesh_data['pressure_data'],
            mesh_data['nut_data'],
            nu,
            mesh_data['cell_volumes'],
            mesh_data['cell_centers'],
            mesh_data['cell_point_ids_flat'],
            mesh_data['cell_point_ids_offsets'],
            mesh_data['neighbors_flat'],
            mesh_data['neighbors_offsets'],
            mesh_data['face_point_ids_flat'],
            mesh_data['face_offsets']
        )
    
    elif device == "gpu":
        import cupy as cp
        
        # Transfer data to GPU
        points = cp.asarray(mesh_data['points'])
        velocity_data = cp.asarray(mesh_data['velocity_data'])
        pressure_data = cp.asarray(mesh_data['pressure_data'])
        nut_data = cp.asarray(mesh_data['nut_data'])
        cell_volumes = cp.asarray(mesh_data['cell_volumes'])
        cell_centers = cp.asarray(mesh_data['cell_centers'])
        cell_point_ids_flat = cp.asarray(mesh_data['cell_point_ids_flat'])
        cell_point_ids_offsets = cp.asarray(mesh_data['cell_point_ids_offsets'])
        neighbors_flat = cp.asarray(mesh_data['neighbors_flat'])
        neighbors_offsets = cp.asarray(mesh_data['neighbors_offsets'])
        face_point_ids_flat = cp.asarray(mesh_data['face_point_ids_flat'])
        face_offsets = cp.asarray(mesh_data['face_offsets'])
        
        # Allocate output arrays on GPU
        continuity = cp.zeros(n_cells, dtype=np.float64)
        momentum_x = cp.zeros(n_cells, dtype=np.float64)
        momentum_y = cp.zeros(n_cells, dtype=np.float64)
        momentum_z = cp.zeros(n_cells, dtype=np.float64)
        
        # Launch kernel
        threads_per_block = 256
        blocks_per_grid = (n_cells + threads_per_block - 1) // threads_per_block
        
        compute_all_residuals_gpu[blocks_per_grid, threads_per_block](
            n_cells, points, velocity_data, pressure_data, nut_data, nu,
            cell_volumes, cell_centers, cell_point_ids_flat, cell_point_ids_offsets,
            neighbors_flat, neighbors_offsets, face_point_ids_flat, face_offsets,
            continuity, momentum_x, momentum_y, momentum_z
        )
        
        # Transfer back to CPU
        continuity = cp.asnumpy(continuity)
        momentum_x = cp.asnumpy(momentum_x)
        momentum_y = cp.asnumpy(momentum_y)
        momentum_z = cp.asnumpy(momentum_z)
    
    else:
        raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'gpu'.")
    
    return continuity, momentum_x, momentum_y, momentum_z


def compute_residuals_fvm(
    filename,
    velocity_field="UMean",
    pressure_field="pMean",
    nut_field="nutMean",
    nu=1.5e-5,
    device="cpu",
    save_name=None,
    progress_bar=False
):
    """
    Compute FVM-based continuity and momentum residuals for a VTU mesh file.
    
    Parameters
    ----------
    filename : str
        Path to the VTU file.
    velocity_field : str
        Name of the velocity field.
    pressure_field : str
        Name of the pressure field.
    nut_field : str
        Name of the turbulent viscosity field.
    nu : float
        Kinematic viscosity.
    device : str
        Device to use: "cpu" or "gpu".
    save_name : str, optional
        If provided, save the results to this file.
    progress_bar : bool
        Whether to show progress bars.
    
    Returns
    -------
    vtk.vtkUnstructuredGrid
        The mesh with residuals added to cell data.
    """
    import time
    
    print("\nLoading mesh...")
    t0 = time.time()
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    ugrid = reader.GetOutput()
    t_loading = time.time() - t0
    print(f"  {ugrid.GetNumberOfCells():,} cells, {ugrid.GetNumberOfPoints():,} points")
    print(f"  Time: {t_loading:.2f} seconds")
    
    print("\nPreparing mesh data...")
    t0 = time.time()
    mesh_data = _prepare_mesh_data(ugrid, velocity_field, pressure_field, nut_field)
    t_preparation = time.time() - t0
    print(f"  Time: {t_preparation:.2f} seconds")
    
    print(f"\nComputing residuals on {device.upper()}...")
    t0 = time.time()
    continuity, momentum_x, momentum_y, momentum_z = _compute_residuals_fvm(mesh_data, nu, device)
    t_computation = time.time() - t0
    print(f"  Time: {t_computation:.2f} seconds")
    
    # Add results to mesh
    ugrid = mesh_data['ugrid']
    continuity_vtk = numpy_support.numpy_to_vtk(continuity)
    continuity_vtk.SetName("Continuity_FVM")
    ugrid.GetCellData().AddArray(continuity_vtk)
    
    momentum_x_vtk = numpy_support.numpy_to_vtk(momentum_x)
    momentum_x_vtk.SetName("Momentum_X_FVM")
    ugrid.GetCellData().AddArray(momentum_x_vtk)
    
    momentum_y_vtk = numpy_support.numpy_to_vtk(momentum_y)
    momentum_y_vtk.SetName("Momentum_Y_FVM")
    ugrid.GetCellData().AddArray(momentum_y_vtk)
    
    momentum_z_vtk = numpy_support.numpy_to_vtk(momentum_z)
    momentum_z_vtk.SetName("Momentum_Z_FVM")
    ugrid.GetCellData().AddArray(momentum_z_vtk)
    
    # Print statistics
    print("\nResidual Statistics:")
    print(f"  Continuity:  min={np.min(continuity):.2e}, max={np.max(continuity):.2e}, mean={np.mean(continuity):.2e}")
    print(f"  Momentum X:  min={np.min(momentum_x):.2e}, max={np.max(momentum_x):.2e}, mean={np.mean(momentum_x):.2e}")
    print(f"  Momentum Y:  min={np.min(momentum_y):.2e}, max={np.max(momentum_y):.2e}, mean={np.mean(momentum_y):.2e}")
    print(f"  Momentum Z:  min={np.min(momentum_z):.2e}, max={np.max(momentum_z):.2e}, mean={np.mean(momentum_z):.2e}")
    
    # Save if requested
    if save_name:
        print(f"\nSaving results to: {save_name}")
        t0 = time.time()
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(save_name)
        writer.SetInputData(ugrid)
        writer.Write()
        t_saving = time.time() - t0
        print(f"  Time: {t_saving:.2f} seconds")
    else:
        t_saving = 0.0
    
    # Print timing summary
    t_total = t_loading + t_preparation + t_computation + t_saving
    print("\n" + "-" * 80)
    print("Timing Summary:")
    print("-" * 80)
    print(f"  Loading mesh:       {t_loading:8.2f} seconds ({t_loading/t_total*100:5.1f}%)")
    print(f"  Preparing data:     {t_preparation:8.2f} seconds ({t_preparation/t_total*100:5.1f}%)")
    print(f"  Computing residuals:{t_computation:8.2f} seconds ({t_computation/t_total*100:5.1f}%)")
    if save_name:
        print(f"  Saving results:     {t_saving:8.2f} seconds ({t_saving/t_total*100:5.1f}%)")
    print(f"  {'-' * 76}")
    print(f"  Total time:         {t_total:8.2f} seconds")
    print("-" * 80)
    
    return ugrid


if __name__ == "__main__":
    # Configuration
    filename = "./internal_419_rans_with_residuals_clipped_10M.vtu"
    nu = 1.5881327800829875e-5
    
    print("=" * 80)
    print("FVM Residuals Computation: CPU vs GPU Comparison")
    print("=" * 80)
    
    # Compute on CPU
    print("\n" + "=" * 80)
    print("COMPUTING ON CPU")
    print("=" * 80)
    ugrid_cpu = compute_residuals_fvm(
        filename,
        velocity_field="UMean",
        pressure_field="pMean",
        nut_field="nutMean",
        nu=nu,
        device="cpu",
        save_name="internal_419_fvm_residuals_cpu.vtu"
    )
    
    # Compute on GPU
    try:
        print("\n" + "=" * 80)
        print("COMPUTING ON GPU")
        print("=" * 80)
        ugrid_gpu = compute_residuals_fvm(
            filename,
            velocity_field="UMean",
            pressure_field="pMean",
            nut_field="nutMean",
            nu=nu,
            device="gpu",
            save_name="internal_419_fvm_residuals_gpu.vtu"
        )
        
        # Compare results
        print("\n" + "=" * 80)
        print("COMPARING CPU vs GPU RESULTS")
        print("=" * 80)
        
        cont_cpu = numpy_support.vtk_to_numpy(ugrid_cpu.GetCellData().GetArray("Continuity_FVM"))
        cont_gpu = numpy_support.vtk_to_numpy(ugrid_gpu.GetCellData().GetArray("Continuity_FVM"))
        
        mom_x_cpu = numpy_support.vtk_to_numpy(ugrid_cpu.GetCellData().GetArray("Momentum_X_FVM"))
        mom_x_gpu = numpy_support.vtk_to_numpy(ugrid_gpu.GetCellData().GetArray("Momentum_X_FVM"))
        
        mom_y_cpu = numpy_support.vtk_to_numpy(ugrid_cpu.GetCellData().GetArray("Momentum_Y_FVM"))
        mom_y_gpu = numpy_support.vtk_to_numpy(ugrid_gpu.GetCellData().GetArray("Momentum_Y_FVM"))
        
        mom_z_cpu = numpy_support.vtk_to_numpy(ugrid_cpu.GetCellData().GetArray("Momentum_Z_FVM"))
        mom_z_gpu = numpy_support.vtk_to_numpy(ugrid_gpu.GetCellData().GetArray("Momentum_Z_FVM"))
        
        print(f"Continuity - Max absolute difference: {np.max(np.abs(cont_cpu - cont_gpu)):.2e}")
        print(f"Continuity - Relative error (RMS): {np.sqrt(np.mean((cont_cpu - cont_gpu)**2)) / (np.std(cont_cpu) + 1e-10):.2e}")
        
        print(f"\nMomentum X - Max absolute difference: {np.max(np.abs(mom_x_cpu - mom_x_gpu)):.2e}")
        print(f"Momentum X - Relative error (RMS): {np.sqrt(np.mean((mom_x_cpu - mom_x_gpu)**2)) / (np.std(mom_x_cpu) + 1e-10):.2e}")
        
        print(f"\nMomentum Y - Max absolute difference: {np.max(np.abs(mom_y_cpu - mom_y_gpu)):.2e}")
        print(f"Momentum Y - Relative error (RMS): {np.sqrt(np.mean((mom_y_cpu - mom_y_gpu)**2)) / (np.std(mom_y_cpu) + 1e-10):.2e}")
        
        print(f"\nMomentum Z - Max absolute difference: {np.max(np.abs(mom_z_cpu - mom_z_gpu)):.2e}")
        print(f"Momentum Z - Relative error (RMS): {np.sqrt(np.mean((mom_z_cpu - mom_z_gpu)**2)) / (np.std(mom_z_cpu) + 1e-10):.2e}")
        
    except Exception as e:
        print(f"\nGPU computation failed: {e}")
        print("This is expected if CUDA is not available.")
    
    print("\n" + "=" * 80)
    print("DONE")
    print("=" * 80)