# Dataset Organization Guide

This guide explains how to organize the HiLiftAeroML processed dataset into train/validation/test splits.

## Dataset Structure

The processed dataset contains files in the format: `geo_<GEO_ID>_AoA_<ANGLE>.npy`

- **GEO_ID**: Geometry identifier (e.g., LHC001, LHC002, F25, F3734)
  - LHC### = Latin Hypercube samples (different aircraft geometries)
  - F### = Specific geometry configurations
- **AoA**: Angle of Attack in degrees (4, 6, 8, 10, 12, 14, 16, 18, 20, 22)

## Current Dataset Statistics

Based on analysis of `/lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big/`:

- **Total files:** 1,504
- **Unique geometries:** 181
- **AoA values:** 10 (from 4° to 22°)
- **Average files per geometry:** 8.31

## Splitting Strategy

The script uses **geometry-based stratified splitting**:

1. All files from the same geometry ID go to the same split
2. This prevents data leakage (no geometry appears in both train and test)
3. Each split maintains representation across all AoA values
4. Default split: 70% train / 15% val / 15% test

### Recommended Split (70/15/15)

- **Train:** ~1,048 files from ~126 geometries
- **Val:** ~228 files from ~27 geometries
- **Test:** ~228 files from ~28 geometries

## Usage

### Step 1: Analyze the Dataset (No Changes)

```bash
python organize_dataset.py \
    --data_path /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big/
```

This will:
- Analyze file distribution
- Show AoA coverage
- Generate split information in `split_info/` directory
- **No files are modified**

### Step 2: Dry Run (Preview Changes)

```bash
python organize_dataset.py \
    --data_path /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big/ \
    --execute \
    --dry_run
```

This will show what would be done without actually modifying files.

### Step 3: Execute Organization

```bash
python organize_dataset.py \
    --data_path /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big/ \
    --execute \
    --method symlink
```

This will create symbolic links organized into subdirectories.

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data_path` | (see above) | Path to processed dataset |
| `--train_ratio` | 0.7 | Fraction for training |
| `--val_ratio` | 0.15 | Fraction for validation |
| `--test_ratio` | 0.15 | Fraction for testing |
| `--method` | symlink | `symlink`, `copy`, or `move` |
| `--seed` | 42 | Random seed for reproducibility |
| `--dry_run` | False | Preview without changes |
| `--execute` | False | Actually perform organization |

## Organization Methods

### Symlink (Recommended)
```bash
--method symlink
```
- Creates symbolic links to original files
- No disk space overhead
- Original files remain in place
- **Recommended for most use cases**

### Copy
```bash
--method copy
```
- Copies files to new directories
- Doubles disk space usage
- Original files remain in place
- Use if you need independent copies

### Move
```bash
--method move
```
- Moves files to new directories
- No disk space overhead
- **Modifies original directory structure**
- Use with caution

## Output Structure

After execution, files will be organized as:

```
HiLiftAeroML-Processed-Big_split/
├── train/
│   ├── geo_LHC001_AoA_4.npy
│   ├── geo_LHC001_AoA_6.npy
│   └── ...
├── val/
│   ├── geo_LHC050_AoA_10.npy
│   └── ...
└── test/
    ├── geo_LHC100_AoA_12.npy
    └── ...
```

## Split Information Files

The script creates several reference files in `split_info/`:

- `split_summary.txt` - Overall statistics
- `train_geometries.txt` - Geometry IDs in training set
- `val_geometries.txt` - Geometry IDs in validation set
- `test_geometries.txt` - Geometry IDs in test set
- `train_files.txt` - All training filenames
- `val_files.txt` - All validation filenames
- `test_files.txt` - All test filenames

## Custom Split Ratios

For different split ratios (e.g., 80/10/10):

```bash
python organize_dataset.py \
    --data_path /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big/ \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1 \
    --execute
```

## Reproducibility

The split is deterministic based on the random seed. To reproduce the exact same split:
- Use the same `--seed` value (default: 42)
- Use the same `--train_ratio`, `--val_ratio`, `--test_ratio`

To create a different split:
```bash
--seed 123
```

## Updating Config Files

After organizing the dataset, update `conf/config.yaml`:

```yaml
data:
  input_dir: /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big_split/train/
  input_dir_val: /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big_split/val/
```

And for testing in `eval` section:
```yaml
eval:
  test_path: /lustre/fs1/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big_split/test/
```

## Verification

After organization, verify the splits:

```bash
# Count files in each split
ls HiLiftAeroML-Processed-Big_split/train/*.npy | wc -l
ls HiLiftAeroML-Processed-Big_split/val/*.npy | wc -l
ls HiLiftAeroML-Processed-Big_split/test/*.npy | wc -l

# Check a specific geometry doesn't appear in multiple splits
grep "LHC001" split_info/train_geometries.txt
grep "LHC001" split_info/val_geometries.txt
grep "LHC001" split_info/test_geometries.txt
```

## Example Workflow

```bash
# 1. Analyze dataset
python organize_dataset.py

# 2. Review split_info/split_summary.txt

# 3. Dry run to preview
python organize_dataset.py --execute --dry_run

# 4. Execute with symlinks (recommended)
python organize_dataset.py --execute --method symlink

# 5. Verify the organization
cat split_info/split_summary.txt
```

