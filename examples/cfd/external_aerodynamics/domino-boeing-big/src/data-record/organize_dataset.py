#!/usr/bin/env python3
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
Script to organize the HiLiftAeroML processed dataset into train/val/test splits.
Splits are based on LHC (Latin Hypercube samples) and AoA (Angle of Attack) values
to ensure good distribution across both parameters.
"""

import os
import re
import shutil
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


def parse_filename(filename):
    """
    Parse LHC ID and AoA from filename.
    Expected format: geo_LHC###_AoA_## or geo_F####_AoA_##
    
    Returns:
        tuple: (geometry_id, aoa_value) or (None, None) if parsing fails
    """
    # Match patterns like geo_LHC001_AoA_12 or geo_F25_AoA_10
    match = re.match(r"geo_([A-Z]+\d+)_AoA_(\d+(?:\.\d+)?)", filename)
    if match:
        geo_id = match.group(1)  # e.g., "LHC001" or "F25"
        aoa = float(match.group(2))
        return geo_id, aoa
    return None, None


def analyze_dataset(data_path):
    """
    Analyze the dataset to understand distribution of LHC samples and AoA values.
    
    Returns:
        dict: Contains file list, unique geo IDs, unique AoA values, and distribution info
    """
    data_path = Path(data_path)
    
    # Find all .npy files
    npy_files = list(data_path.glob("*.npy"))
    
    print(f"\n{'=' * 80}")
    print(f"DATASET ANALYSIS")
    print(f"{'=' * 80}")
    print(f"Data path: {data_path}")
    print(f"Total .npy files found: {len(npy_files)}")
    
    # Parse all filenames
    geo_aoa_map = defaultdict(list)  # geo_id -> list of AoA values
    aoa_geo_map = defaultdict(list)  # aoa -> list of geo_ids
    valid_files = []
    
    for npy_file in npy_files:
        filename = npy_file.stem  # Get filename without extension
        geo_id, aoa = parse_filename(filename)
        
        if geo_id is not None and aoa is not None:
            geo_aoa_map[geo_id].append(aoa)
            aoa_geo_map[aoa].append(geo_id)
            valid_files.append((filename, geo_id, aoa))
    
    print(f"Valid parsed files: {len(valid_files)}")
    
    # Get unique values
    unique_geo_ids = sorted(geo_aoa_map.keys())
    unique_aoa_values = sorted(aoa_geo_map.keys())
    
    print(f"\nUnique geometry IDs (LHC samples): {len(unique_geo_ids)}")
    print(f"Geometry IDs: {unique_geo_ids[:10]}... (showing first 10)")
    
    print(f"\nUnique AoA values: {len(unique_aoa_values)}")
    print(f"AoA values: {unique_aoa_values}")
    
    # Distribution analysis
    print(f"\n{'=' * 80}")
    print(f"DISTRIBUTION ANALYSIS")
    print(f"{'=' * 80}")
    
    # Count files per geometry ID
    geo_counts = {geo_id: len(aoas) for geo_id, aoas in geo_aoa_map.items()}
    print(f"\nFiles per geometry ID:")
    print(f"  Min: {min(geo_counts.values())}")
    print(f"  Max: {max(geo_counts.values())}")
    print(f"  Mean: {np.mean(list(geo_counts.values())):.2f}")
    
    # Count files per AoA
    aoa_counts = {aoa: len(geos) for aoa, geos in aoa_geo_map.items()}
    print(f"\nFiles per AoA value:")
    print(f"  Min: {min(aoa_counts.values())}")
    print(f"  Max: {max(aoa_counts.values())}")
    print(f"  Mean: {np.mean(list(aoa_counts.values())):.2f}")
    
    # Show distribution per AoA
    print(f"\nDetailed AoA distribution:")
    for aoa in unique_aoa_values:
        print(f"  AoA {aoa:5.1f}: {aoa_counts[aoa]:4d} samples")
    
    return {
        'valid_files': valid_files,
        'unique_geo_ids': unique_geo_ids,
        'unique_aoa_values': unique_aoa_values,
        'geo_aoa_map': geo_aoa_map,
        'aoa_geo_map': aoa_geo_map,
        'geo_counts': geo_counts,
        'aoa_counts': aoa_counts,
    }


def create_stratified_split(dataset_info, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
    """
    Create stratified split ensuring good distribution of both LHC samples and AoA values.
    
    Strategy:
    1. Split geometry IDs (LHC samples) into train/val/test
    2. For each geometry ID, all its AoA values go to the same split
    3. This ensures no data leakage across geometries
    
    Args:
        dataset_info: Dictionary from analyze_dataset
        train_ratio: Fraction for training (default 0.7)
        val_ratio: Fraction for validation (default 0.15)
        test_ratio: Fraction for testing (default 0.15)
        random_seed: Random seed for reproducibility
        
    Returns:
        dict: Contains train_files, val_files, test_files lists
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    np.random.seed(random_seed)
    
    # Get all geometry IDs and shuffle them
    geo_ids = dataset_info['unique_geo_ids'].copy()
    np.random.shuffle(geo_ids)
    
    n_total = len(geo_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split geometry IDs
    train_geo_ids = set(geo_ids[:n_train])
    val_geo_ids = set(geo_ids[n_train:n_train + n_val])
    test_geo_ids = set(geo_ids[n_train + n_val:])
    
    print(f"\n{'=' * 80}")
    print(f"SPLIT STRATEGY")
    print(f"{'=' * 80}")
    print(f"Random seed: {random_seed}")
    print(f"Split ratios: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    print(f"\nGeometry ID splits:")
    print(f"  Train geometries: {len(train_geo_ids)}")
    print(f"  Val geometries:   {len(val_geo_ids)}")
    print(f"  Test geometries:  {len(test_geo_ids)}")
    
    # Assign files to splits
    train_files = []
    val_files = []
    test_files = []
    
    for filename, geo_id, aoa in dataset_info['valid_files']:
        if geo_id in train_geo_ids:
            train_files.append((filename, geo_id, aoa))
        elif geo_id in val_geo_ids:
            val_files.append((filename, geo_id, aoa))
        elif geo_id in test_geo_ids:
            test_files.append((filename, geo_id, aoa))
    
    print(f"\nFile splits:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files:   {len(val_files)}")
    print(f"  Test files:  {len(test_files)}")
    
    # Analyze AoA distribution in each split
    train_aoas = defaultdict(int)
    val_aoas = defaultdict(int)
    test_aoas = defaultdict(int)
    
    for _, _, aoa in train_files:
        train_aoas[aoa] += 1
    for _, _, aoa in val_files:
        val_aoas[aoa] += 1
    for _, _, aoa in test_files:
        test_aoas[aoa] += 1
    
    print(f"\n{'=' * 80}")
    print(f"AoA DISTRIBUTION PER SPLIT")
    print(f"{'=' * 80}")
    print(f"{'AoA':>6} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Total':>8}")
    print(f"{'-' * 80}")
    
    all_aoas = sorted(dataset_info['unique_aoa_values'])
    for aoa in all_aoas:
        train_count = train_aoas.get(aoa, 0)
        val_count = val_aoas.get(aoa, 0)
        test_count = test_aoas.get(aoa, 0)
        total_count = train_count + val_count + test_count
        print(f"{aoa:6.1f} | {train_count:8d} | {val_count:8d} | {test_count:8d} | {total_count:8d}")
    
    return {
        'train_files': train_files,
        'val_files': val_files,
        'test_files': test_files,
        'train_geo_ids': train_geo_ids,
        'val_geo_ids': val_geo_ids,
        'test_geo_ids': test_geo_ids,
    }


def organize_files(data_path, split_info, method='move', dry_run=True, inplace=True):
    """
    Organize files into train/val/test directories.
    
    Args:
        data_path: Path to the processed dataset
        split_info: Dictionary from create_stratified_split
        method: 'symlink' (create symbolic links) or 'copy' (copy files) or 'move' (move files)
        dry_run: If True, only print what would be done without doing it
        inplace: If True, create train/val/test inside data_path. If False, create separate _split directory
    """
    data_path = Path(data_path)
    
    # Create split directories
    if inplace:
        # Create train/val/test folders inside the data directory
        train_dir = data_path / "train"
        val_dir = data_path / "val"
        test_dir = data_path / "test"
    else:
        # Create separate split directory (old behavior)
        train_dir = data_path.parent / f"{data_path.name}_split" / "train"
        val_dir = data_path.parent / f"{data_path.name}_split" / "val"
        test_dir = data_path.parent / f"{data_path.name}_split" / "test"
    
    print(f"\n{'=' * 80}")
    print(f"FILE ORGANIZATION")
    print(f"{'=' * 80}")
    print(f"Source directory: {data_path}")
    print(f"Target directories:")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")
    print(f"  Test:  {test_dir}")
    print(f"Method: {method}")
    print(f"Dry run: {dry_run}")
    
    if dry_run:
        print(f"\n*** DRY RUN MODE - No files will be modified ***\n")
    else:
        # Create directories
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nCreated output directories")
    
    # Process each split
    splits = [
        ('train', split_info['train_files'], train_dir),
        ('val', split_info['val_files'], val_dir),
        ('test', split_info['test_files'], test_dir),
    ]
    
    for split_name, file_list, target_dir in splits:
        print(f"\n{split_name.upper()} split: {len(file_list)} files")
        
        for idx, (filename, geo_id, aoa) in enumerate(file_list):
            src_file = data_path / f"{filename}.npy"
            dst_file = target_dir / f"{filename}.npy"
            
            if not src_file.exists():
                print(f"  WARNING: Source file not found: {src_file}")
                continue
            
            if dry_run:
                if idx < 5:  # Show first 5 examples
                    print(f"  Would {method}: {filename}.npy")
            else:
                try:
                    if method == 'symlink':
                        if dst_file.exists() or dst_file.is_symlink():
                            dst_file.unlink()
                        dst_file.symlink_to(src_file)
                    elif method == 'copy':
                        shutil.copy2(src_file, dst_file)
                    elif method == 'move':
                        shutil.move(src_file, dst_file)
                    else:
                        raise ValueError(f"Unknown method: {method}")
                    
                    if (idx + 1) % 100 == 0:
                        print(f"  Processed {idx + 1}/{len(file_list)} files...")
                except Exception as e:
                    print(f"  ERROR processing {filename}: {e}")
        
        if not dry_run:
            print(f"  Completed {split_name} split: {len(file_list)} files")


def save_split_info(output_path, split_info, dataset_info):
    """Save split information to text files for reference."""
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save geometry ID assignments
    with open(output_path / "train_geometries.txt", 'w') as f:
        for geo_id in sorted(split_info['train_geo_ids']):
            f.write(f"{geo_id}\n")
    
    with open(output_path / "val_geometries.txt", 'w') as f:
        for geo_id in sorted(split_info['val_geo_ids']):
            f.write(f"{geo_id}\n")
    
    with open(output_path / "test_geometries.txt", 'w') as f:
        for geo_id in sorted(split_info['test_geo_ids']):
            f.write(f"{geo_id}\n")
    
    # Save file lists
    with open(output_path / "train_files.txt", 'w') as f:
        for filename, geo_id, aoa in split_info['train_files']:
            f.write(f"{filename}\n")
    
    with open(output_path / "val_files.txt", 'w') as f:
        for filename, geo_id, aoa in split_info['val_files']:
            f.write(f"{filename}\n")
    
    with open(output_path / "test_files.txt", 'w') as f:
        for filename, geo_id, aoa in split_info['test_files']:
            f.write(f"{filename}\n")
    
    # Save summary statistics
    with open(output_path / "split_summary.txt", 'w') as f:
        f.write(f"Dataset Split Summary\n")
        f.write(f"{'=' * 80}\n\n")
        
        f.write(f"Total files: {len(dataset_info['valid_files'])}\n")
        f.write(f"Unique geometries: {len(dataset_info['unique_geo_ids'])}\n")
        f.write(f"Unique AoA values: {len(dataset_info['unique_aoa_values'])}\n\n")
        
        f.write(f"Split sizes:\n")
        f.write(f"  Train: {len(split_info['train_files'])} files, {len(split_info['train_geo_ids'])} geometries\n")
        f.write(f"  Val:   {len(split_info['val_files'])} files, {len(split_info['val_geo_ids'])} geometries\n")
        f.write(f"  Test:  {len(split_info['test_files'])} files, {len(split_info['test_geo_ids'])} geometries\n\n")
        
        f.write(f"AoA distribution:\n")
        f.write(f"{'AoA':>6} | {'Train':>8} | {'Val':>8} | {'Test':>8} | {'Total':>8}\n")
        f.write(f"{'-' * 60}\n")
        
        train_aoas = defaultdict(int)
        val_aoas = defaultdict(int)
        test_aoas = defaultdict(int)
        
        for _, _, aoa in split_info['train_files']:
            train_aoas[aoa] += 1
        for _, _, aoa in split_info['val_files']:
            val_aoas[aoa] += 1
        for _, _, aoa in split_info['test_files']:
            test_aoas[aoa] += 1
        
        for aoa in sorted(dataset_info['unique_aoa_values']):
            train_count = train_aoas.get(aoa, 0)
            val_count = val_aoas.get(aoa, 0)
            test_count = test_aoas.get(aoa, 0)
            total_count = train_count + val_count + test_count
            f.write(f"{aoa:6.1f} | {train_count:8d} | {val_count:8d} | {test_count:8d} | {total_count:8d}\n")
    
    print(f"\nSplit information saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Organize HiLiftAeroML processed dataset into train/val/test splits'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big/',
        help='Path to the processed dataset directory'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='Fraction of data for training (default: 0.7)'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='Fraction of data for validation (default: 0.15)'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='Fraction of data for testing (default: 0.15)'
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['symlink', 'copy', 'move'],
        default='move',
        help='Method to organize files: symlink, copy, or move (default: move)'
    )
    parser.add_argument(
        '--inplace',
        action='store_true',
        default=True,
        help='Create train/val/test folders inside data_path (default: True)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Perform a dry run without actually modifying files'
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Execute the file organization (without this, only analysis is performed)'
    )
    
    args = parser.parse_args()
    
    # Validate data path exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"ERROR: Data path does not exist: {data_path}")
        return
    
    # Step 1: Analyze dataset
    dataset_info = analyze_dataset(args.data_path)
    
    if len(dataset_info['valid_files']) == 0:
        print("\nERROR: No valid .npy files found in the dataset!")
        return
    
    # Step 2: Create split
    split_info = create_stratified_split(
        dataset_info,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_seed=args.seed
    )
    
    # Step 3: Save split information
    script_dir = Path(__file__).parent
    save_split_info(script_dir / "split_info", split_info, dataset_info)
    
    # Step 4: Organize files if requested
    if args.execute:
        print(f"\n{'=' * 80}")
        print(f"EXECUTING FILE ORGANIZATION")
        print(f"{'=' * 80}")
        
        organize_files(
            args.data_path,
            split_info,
            method=args.method,
            dry_run=args.dry_run,
            inplace=args.inplace
        )
        
        if not args.dry_run:
            print(f"\n{'=' * 80}")
            print(f"ORGANIZATION COMPLETE")
            print(f"{'=' * 80}")
            
            if args.inplace:
                print(f"\nDataset has been organized into:")
                print(f"  Train: {data_path / 'train'}")
                print(f"  Val:   {data_path / 'val'}")
                print(f"  Test:  {data_path / 'test'}")
            else:
                print(f"\nDataset has been organized into:")
                print(f"  Train: {data_path.parent / f'{data_path.name}_split' / 'train'}")
                print(f"  Val:   {data_path.parent / f'{data_path.name}_split' / 'val'}")
                print(f"  Test:  {data_path.parent / f'{data_path.name}_split' / 'test'}")
        else:
            print(f"\n{'=' * 80}")
            print(f"DRY RUN COMPLETE")
            print(f"{'=' * 80}")
            print(f"\nTo execute the organization, run again with --execute flag")
            print(f"To execute without dry run, use: --execute (without --dry_run)")
    else:
        print(f"\n{'=' * 80}")
        print(f"ANALYSIS COMPLETE")
        print(f"{'=' * 80}")
        print(f"\nTo organize the files, run again with --execute flag")
        print(f"Example commands:")
        print(f"  Dry run:  python organize_dataset.py --execute --dry_run")
        print(f"  Execute:  python organize_dataset.py --execute")


if __name__ == "__main__":
    main()

