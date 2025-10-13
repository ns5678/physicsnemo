#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Quick script to organize the HiLiftAeroML dataset
# Edit the DATA_PATH variable below to match your dataset location

DATA_PATH="/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big/"

echo "=========================================="
echo "HiLiftAeroML Dataset Organization"
echo "=========================================="
echo ""

# Check if user wants to execute
if [ "$1" == "--execute" ]; then
    echo "EXECUTING: Moving files into train/val/test folders..."
    python organize_dataset.py \
        --data_path "$DATA_PATH" \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --method move \
        --inplace \
        --seed 42 \
        --execute
    
    echo ""
    echo "Organization complete!"
    echo "Files organized in: $DATA_PATH"
    echo ""
    echo "Update your config.yaml with:"
    echo "  data:"
    echo "    input_dir: ${DATA_PATH%/}/train/"
    echo "    input_dir_val: ${DATA_PATH%/}/val/"
    
elif [ "$1" == "--dry-run" ]; then
    echo "DRY RUN: Previewing what would be done..."
    python organize_dataset.py \
        --data_path "$DATA_PATH" \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --method move \
        --inplace \
        --seed 42 \
        --execute \
        --dry_run

else
    echo "ANALYSIS MODE: Analyzing dataset without modifying files..."
    python organize_dataset.py \
        --data_path "$DATA_PATH" \
        --train_ratio 0.7 \
        --val_ratio 0.15 \
        --test_ratio 0.15 \
        --seed 42
    
    echo ""
    echo "=========================================="
    echo "Next steps:"
    echo "=========================================="
    echo "1. Review the split_info/split_summary.txt file"
    echo "2. Run with --dry-run to preview: ./organize_dataset.sh --dry-run"
    echo "3. Run with --execute to organize: ./organize_dataset.sh --execute"
fi

