import os
from pathlib import Path

# Define the base directory
base_dir = Path("/lustre/fsw/portfolios/coreai/projects/coreai_modulus_cae/datasets/HiLiftAeroML-Processed-Big")

# Get train folder
train_dir = base_dir / "train"
train_files = sorted([f.name for f in train_dir.glob("*.npy")])

# Get val folder
val_dir = base_dir / "val"
val_files = sorted([f.name for f in val_dir.glob("*.npy")])


train_output = Path("train_files.txt")
with open(train_output, "w") as f:
    f.write("\n".join(train_files))


val_output = Path("val_files.txt")
with open(val_output, "w") as f:
    f.write("\n".join(val_files))
