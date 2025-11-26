#!/usr/bin/env python3
"""Compare and plot training logs from multiple experiments."""

import re
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np


def extract_label_from_path(paths):
    """Extract unique labels from file paths by finding the differing part."""
    paths = [Path(p) for p in paths]
    
    if len(paths) == 1:
        # For single path, use the parent directory name
        return [paths[0].parent.name]
    
    # Split all paths into parts
    all_parts = [p.parts for p in paths]
    
    # Find the first position where paths differ
    labels = []
    for parts in all_parts:
        # Try to find a unique identifier (typically a directory name)
        for i, part in enumerate(parts):
            # Check if this part is different across paths
            if any(other[i] != part for other in all_parts if len(other) > i):
                labels.append(part)
                break
        else:
            # If no difference found, use the parent directory
            labels.append(paths[labels.__len__()].parent.name)
    
    return labels


def parse_log_file(log_path):
    """Parse training log and extract validation metrics."""
    data = defaultdict(list)
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Find all validation sections
    val_pattern = r'Epoch (\d+) VALIDATION Average Metrics:\n.*?\n(.*?)\n\n'
    loss_pattern = r'Epoch (\d+) VALIDATION Loss Components:\n.*?\n(.*?)\n\n'
    
    # Extract L2 metrics
    for match in re.finditer(val_pattern, content, re.DOTALL):
        epoch = int(match.group(1))
        metrics_text = match.group(2)
        
        # Parse each metric line - only lines with alphabetic metric names
        metric_lines = re.findall(r'\|\s*([a-zA-Z_]\w*)\s*\|\s*([\d.e+-]+)\s*\|', metrics_text)
        
        data['epoch'].append(epoch)
        for metric_name, value in metric_lines:
            metric_name = metric_name.strip()
            data[metric_name].append(float(value))
    
    # Extract loss components
    epoch_to_idx = {e: i for i, e in enumerate(data['epoch'])}
    for match in re.finditer(loss_pattern, content, re.DOTALL):
        epoch = int(match.group(1))
        if epoch not in epoch_to_idx:
            continue
            
        idx = epoch_to_idx[epoch]
        loss_text = match.group(2)
        
        # Parse each loss line - only lines with alphabetic loss names
        loss_lines = re.findall(r'\|\s*([a-zA-Z_]\w*)\s*\|\s*([\d.e+-]+)\s*\|', loss_text)
        
        for loss_name, value in loss_lines:
            loss_name = loss_name.strip()
            if len(data[loss_name]) <= idx:
                data[loss_name].extend([np.nan] * (idx + 1 - len(data[loss_name])))
            data[loss_name][idx] = float(value)
    
    return data


def plot_comparison(log_files, output_path=None):
    """Create comparison plots for multiple log files."""
    # Parse all log files
    all_data = []
    labels = extract_label_from_path(log_files)
    
    for log_file in log_files:
        data = parse_log_file(log_file)
        all_data.append(data)
    
    # Determine metrics to plot
    l2_metrics = ['l2_vol_pressure', 'l2_velocity_x', 'l2_velocity_y', 'l2_velocity_z', 'l2_nut']
    loss_metrics = ['loss_vol', 'loss_continuity', 'loss_momentum_x', 'loss_momentum_y', 'loss_momentum_z']
    
    # Filter to only include metrics present in data
    l2_metrics = [m for m in l2_metrics if any(m in data for data in all_data)]
    loss_metrics = [m for m in loss_metrics if any(m in data for data in all_data)]
    
    # Create subplots
    n_plots = len(l2_metrics) + len(loss_metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots))
    
    if n_plots == 1:
        axes = [axes]
    
    # Plot L2 metrics
    for i, metric in enumerate(l2_metrics):
        ax = axes[i]
        for data, label in zip(all_data, labels):
            if metric in data and len(data[metric]) > 0:
                ax.semilogy(data['epoch'], data[metric], label=label, alpha=0.7)
                # ax.plot(data['epoch'], data[metric], label=label, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} (Validation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot loss components
    for i, metric in enumerate(loss_metrics):
        ax = axes[len(l2_metrics) + i]
        for data, label in zip(all_data, labels):
            if metric in data and len(data[metric]) > 0:
                ax.semilogy(data['epoch'], data[metric], label=label, alpha=0.7)
                # ax.plot(data['epoch'], data[metric], label=label, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} (Validation)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path is None:
        output_path = 'comparison.png'
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare training logs')
    parser.add_argument('log_files', nargs='+', help='Training log files to compare')
    parser.add_argument('-o', '--output', help='Output plot file path (optional)')
    
    args = parser.parse_args()
    
    plot_comparison(args.log_files, args.output)


if __name__ == '__main__':
    main()

