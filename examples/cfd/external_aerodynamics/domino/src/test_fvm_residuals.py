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
Unit tests for FVM residual computation.

Tests include:
1. CPU vs CUDA vs Warp (full mesh) comparison
2. Full mesh vs batched computation comparison
"""

import unittest
import numpy as np
import time
import sys
import os
import vtk
from vtk.util import numpy_support

from compure_physics_loss_standalone import (
    _prepare_mesh_data,
    compute_all_residuals_cpu,
    compute_all_residuals_gpu
)
from fvm_residuals_warp import (
    compute_residuals_warp_full,
    compute_residuals_warp_batch,
    compute_residuals_warp_full_batched,
    extract_batched_mesh_data,
    compute_residuals_warp_prebatched,
    compute_residuals_warp_cell_centered
)

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    print("WARNING: CuPy not available, skipping CUDA tests")

try:
    import warp as wp
    wp.init()
    WARP_AVAILABLE = True
except ImportError:
    WARP_AVAILABLE = False
    print("WARNING: Warp not available, skipping Warp tests")


class TestFVMResiduals(unittest.TestCase):
    """Test suite for FVM residual computations."""
    
    @classmethod
    def setUpClass(cls):
        """Load test mesh data once for all tests."""
        print("\n" + "="*80)
        print("Setting up test data...")
        print("="*80)
        
        # VTU file path here
        cls.test_mesh_file = "../../../../../../../../domino-phy-informed-dev/dataset_rans/drivaerml_rans/run_1/volume_1.vtu"
        cls.nu = 1.5881327800829875e-5
        cls.output_dir = os.path.dirname(os.path.abspath(__file__))
        
        try:
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(cls.test_mesh_file)
            reader.Update()
            cls.ugrid = reader.GetOutput()  # Store original ugrid
            
            print(f"Loaded mesh: {cls.ugrid.GetNumberOfCells():,} cells, "
                  f"{cls.ugrid.GetNumberOfPoints():,} points")
            
            # Prepare mesh data
            cls.mesh_data = _prepare_mesh_data(
                cls.ugrid,
                velocity_field="UMean",
                pressure_field="pMean",
                nut_field="nutMean"
            )
            
            cls.n_cells = len(cls.mesh_data['cell_volumes'])
            print(f"Prepared mesh data successfully")
            
        except Exception as e:
            print(f"WARNING: Could not load test mesh: {e}")
            print("Tests will be skipped if no mesh is available")
            cls.mesh_data = None
            cls.ugrid = None
            cls.n_cells = 0
    
    @staticmethod
    def save_results_to_vtu(ugrid, continuity, momentum_x, momentum_y, momentum_z, 
                           output_path, method_name=""):
        """
        Save residual results to VTU file for ParaView visualization.
        
        Args:
            ugrid: VTK unstructured grid
            continuity: Continuity residuals [n_cells]
            momentum_x: X-momentum residuals [n_cells]
            momentum_y: Y-momentum residuals [n_cells]
            momentum_z: Z-momentum residuals [n_cells]
            output_path: Output file path
            method_name: Name of method (e.g., "Warp", "CUDA", "CPU")
        """
        # Add residuals as cell data
        suffix = f"_{method_name}" if method_name else ""
        
        continuity_vtk = numpy_support.numpy_to_vtk(continuity)
        continuity_vtk.SetName(f"Continuity_FVM{suffix}")
        ugrid.GetCellData().AddArray(continuity_vtk)
        
        momentum_x_vtk = numpy_support.numpy_to_vtk(momentum_x)
        momentum_x_vtk.SetName(f"Momentum_X_FVM{suffix}")
        ugrid.GetCellData().AddArray(momentum_x_vtk)
        
        momentum_y_vtk = numpy_support.numpy_to_vtk(momentum_y)
        momentum_y_vtk.SetName(f"Momentum_Y_FVM{suffix}")
        ugrid.GetCellData().AddArray(momentum_y_vtk)
        
        momentum_z_vtk = numpy_support.numpy_to_vtk(momentum_z)
        momentum_z_vtk.SetName(f"Momentum_Z_FVM{suffix}")
        ugrid.GetCellData().AddArray(momentum_z_vtk)
        
        # Compute magnitude for easier visualization
        momentum_mag = np.sqrt(momentum_x**2 + momentum_y**2 + momentum_z**2)
        momentum_mag_vtk = numpy_support.numpy_to_vtk(momentum_mag)
        momentum_mag_vtk.SetName(f"Momentum_Magnitude_FVM{suffix}")
        ugrid.GetCellData().AddArray(momentum_mag_vtk)
        
        # Write to file
        writer = vtk.vtkXMLUnstructuredGridWriter()
        writer.SetFileName(output_path)
        writer.SetInputData(ugrid)
        writer.Write()
        
        print(f"Saved results to: {output_path}")
        print(f"  Added arrays: Continuity_FVM{suffix}, Momentum_[XYZ]_FVM{suffix}, "
              f"Momentum_Magnitude_FVM{suffix}")
    
    def setUp(self):
        """Check if mesh data is available before each test."""
        if self.mesh_data is None:
            self.skipTest("Test mesh not available")
        
    def test_08_cell_centered_fvm(self):
        """Test cell-centered FVM (for neural network outputs)."""
        if not WARP_AVAILABLE:
            self.skipTest("Warp not available")
        
        print("\n" + "-"*80)
        print("TEST 8: Cell-Centered FVM (for Neural Network Inference)")
        print("-"*80)
        
        # Process ALL cells in batches
        batch_size = 8192
        print(f"Computing residuals for ALL {self.n_cells:,} cells using cell-centered FVM...")
        print(f"(Face values interpolated between cells, not from points)")
        
        # Initialize full arrays
        continuity_cell = np.zeros(self.n_cells, dtype=np.float64)
        momentum_x_cell = np.zeros(self.n_cells, dtype=np.float64)
        momentum_y_cell = np.zeros(self.n_cells, dtype=np.float64)
        momentum_z_cell = np.zeros(self.n_cells, dtype=np.float64)
        
        # Process in batches
        n_batches = (self.n_cells + batch_size - 1) // batch_size
        print(f"Processing {n_batches} batches...")
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, self.n_cells)
            batch_indices = np.arange(start_idx, end_idx, dtype=np.int32)
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx+1}/{n_batches}: cells {start_idx:,} to {end_idx:,}")
            
            # Extract batched mesh data
            batched_mesh_data = extract_batched_mesh_data(self.mesh_data, batch_indices)
            
            # Compute residuals using cell-centered method
            cont_batch, mom_x_batch, mom_y_batch, mom_z_batch = \
                compute_residuals_warp_cell_centered(batched_mesh_data, self.nu)
            
            # Fill in the results
            continuity_cell[batch_indices] = cont_batch
            momentum_x_cell[batch_indices] = mom_x_batch
            momentum_y_cell[batch_indices] = mom_y_batch
            momentum_z_cell[batch_indices] = mom_z_batch
        
        print(f"✓ All {self.n_cells:,} cells computed successfully")
        
        # Save to VTU
        output_path = os.path.join(self.output_dir, "residuals_fvm_prebatched_cell_based_full.vtu")
        
        # Save using the existing method
        self.save_results_to_vtu(
            self.ugrid,
            continuity_cell,
            momentum_x_cell,
            momentum_y_cell,
            momentum_z_cell,
            output_path,
            method_name="CellBased_Full"
        )
        
        print(f"\n✅ Results saved to: {output_path}")
        print(f"\nResidual Statistics (Cell-Centered FVM):")
        print(f"  Continuity - min: {continuity_cell.min():.2e}, max: {continuity_cell.max():.2e}, "
              f"mean: {continuity_cell.mean():.2e}")
        print(f"  Momentum X - min: {momentum_x_cell.min():.2e}, max: {momentum_x_cell.max():.2e}, "
              f"mean: {momentum_x_cell.mean():.2e}")
        print(f"  Momentum Y - min: {momentum_y_cell.min():.2e}, max: {momentum_y_cell.max():.2e}, "
              f"mean: {momentum_y_cell.mean():.2e}")
        print(f"  Momentum Z - min: {momentum_z_cell.min():.2e}, max: {momentum_z_cell.max():.2e}, "
              f"mean: {momentum_z_cell.mean():.2e}")
                
        # Verify the saved file exists
        self.assertTrue(os.path.exists(output_path), 
                       f"Output file not created: {output_path}")
        
        print("✅ Cell-centered FVM results saved to VTU successfully!")


def run_tests(test_mesh_file=None):
    """
    Run all FVM residual tests.
    
    Args:
        test_mesh_file: Path to test VTU mesh file (optional)
    """
    if test_mesh_file:
        TestFVMResiduals.test_mesh_file = test_mesh_file
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFVMResiduals)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.wasSuccessful():
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FVM residual computations")
    parser.add_argument("--mesh", type=str, default=None,
                       help="Path to test VTU mesh file")
    parser.add_argument("--test", type=str, default=None,
                       help="Specific test to run (e.g., test_01_cpu_computation)")
    
    args = parser.parse_args()
    
    if args.test:
        # Run specific test
        suite = unittest.TestSuite()
        suite.addTest(TestFVMResiduals(args.test))
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        # Run all tests
        success = run_tests(args.mesh)
        sys.exit(0 if success else 1)

