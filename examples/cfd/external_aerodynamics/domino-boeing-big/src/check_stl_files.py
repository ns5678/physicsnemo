#!/usr/bin/env python3
"""
Script to check for missing STL files in directories.
Scans /lustre/users/nashton/cadence/HiLiftAeroML and identifies folders without STL files.
"""

import os
from pathlib import Path

def check_stl_files(base_dir, exclude_folders=None):
    """
    Check for STL files in subdirectories.
    
    Args:
        base_dir: Base directory to scan
        exclude_folders: List of folder names to exclude
    
    Returns:
        tuple: (folders_with_stl, folders_without_stl)
    """
    if exclude_folders is None:
        exclude_folders = []
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return [], []
    
    folders_with_stl = []
    folders_without_stl = []
    
    # Iterate through all subdirectories
    for item in sorted(base_path.iterdir()):
        # Skip if not a directory
        if not item.is_dir():
            continue
        
        # Skip excluded folders
        if item.name in exclude_folders:
            print(f"Skipping excluded folder: {item.name}")
            continue
        
        # Check for STL files in this directory
        stl_files = list(item.glob("*.stl")) + list(item.glob("*.STL"))
        
        if stl_files:
            folders_with_stl.append(item.name)
        else:
            folders_without_stl.append(item.name)
    
    return folders_with_stl, folders_without_stl


def main():
    base_directory = "/lustre/users/nashton/cadence/HiLiftAeroML"
    exclude = ["sample"]
    
    print(f"Scanning directory: {base_directory}")
    print(f"Excluding folders: {exclude}")
    print("=" * 80)
    print()
    
    folders_with, folders_without = check_stl_files(base_directory, exclude)
    
    print(f"FOLDERS WITH STL FILES")
    print("=" * 80)
    print(f"Total: {len(folders_with)}")
    print("=" * 80)
    print()
    for folder in folders_with:
        print(folder)
    
    print()
    print()
    print(f"FOLDERS MISSING STL FILES")
    print("=" * 80)
    print(f"Total: {len(folders_without)}")
    print("=" * 80)
    print()
    for folder in folders_without:
        print(folder)
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total folders scanned: {len(folders_with) + len(folders_without)}")
    print(f"Folders with STL files: {len(folders_with)}")
    print(f"Folders without STL files: {len(folders_without)}")
    
    # Save results to file
    output_file = "stl_check_results.txt"
    with open(output_file, 'w') as f:
        f.write(f"FOLDERS MISSING STL FILES\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total: {len(folders_without)}\n")
        f.write("=" * 80 + "\n\n")
        for folder in folders_without:
            f.write(folder + "\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

