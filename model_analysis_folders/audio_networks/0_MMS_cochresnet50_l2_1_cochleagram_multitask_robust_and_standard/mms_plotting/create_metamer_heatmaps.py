#!/usr/bin/env python3
"""
Script to create heatmaps from merged metamer pickle files.

This script processes the merged pickle files with consistent naming to generate
R² and MSE heatmaps across different dimensions (layers, sounds, runs, etc.).

Usage:
    python create_metamer_heatmaps.py [--pickle_dir merged_metamers] [--output_dir heatmaps]
"""

import argparse
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr, spearmanr

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

# Define the layer order for sorting (matching the original script)
METAMER_LAYERS = [
    'input_after_preproc',
    'conv1',
    'conv1_relu1',
    'layer1',
    'layer2',
    'layer3',
    'layer4_intermediate_layer_1',
    'layer4_intermediate_layer_2',
    'layer4',
    'avgpool',
]

def parse_filename(filename: str) -> Dict[str, str]:
    """
    Parse the consistent filename format to extract parameters.
    
    Expected format:
    run_{run_number}_{experiment_dir}_SOUNDID_{sound_id}_SUBCLIP{subclip_idx}.pckl
    
    Returns a dictionary with parsed parameters.
    """
    # Remove .pckl extension
    name = filename.replace('.pckl', '')
    
    # Extract run number
    run_match = re.match(r'run_(\d+)_', name)
    run_number = run_match.group(1) if run_match else "unknown"
    
    # Extract sound ID
    sound_match = re.search(r'SOUNDID_(\d+)_', name)
    sound_id = sound_match.group(1) if sound_match else "unknown"
    
    # Extract subclip
    subclip_match = re.search(r'SUBCLIP(\d+)', name)
    subclip_idx = subclip_match.group(1) if subclip_match else "0"
    
    # Extract model type (ROBUST or STANDARD)
    model_match = re.search(r'_(ROBUST|STANDARD)_SOUNDID', name)
    model_type = model_match.group(1) if model_match else "unknown"
    
    # Extract random seed
    seed_match = re.search(r'_RS(\d+)_', name)
    random_seed = seed_match.group(1) if seed_match else "unknown"
    
    # Extract iterations
    iter_match = re.search(r'_I(\d+)_', name)
    iterations = iter_match.group(1) if iter_match else "unknown"
    
    # Extract learning rate
    lr_match = re.search(r'_LR([\d.]+)_', name)
    learning_rate = lr_match.group(1) if lr_match else "unknown"
    
    # Extract LR decay
    decay_match = re.search(r'_DECAY([\d.]+)_', name)
    lr_decay = decay_match.group(1) if decay_match else "unknown"
    
    return {
        'run_number': run_number,
        'sound_id': sound_id,
        'subclip_idx': subclip_idx,
        'model_type': model_type,
        'random_seed': random_seed,
        'iterations': iterations,
        'learning_rate': learning_rate,
        'lr_decay': lr_decay,
        'filename': filename
    }

def squared_mean_activations(layer: str, acts: torch.Tensor) -> Tuple[Optional[torch.Tensor], int]:
    """Compute squared mean activations for correlation analysis."""
    if acts is None or isinstance(acts, dict):
        return None, 0
    if "final" in layer:
        if len(acts.shape) >= 3:
            return acts, acts.shape[1] * acts.shape[2]
        return acts, acts.numel()
    mean_sq = torch.mean(acts ** 2, dim=-1)
    if len(mean_sq.shape) >= 3:
        return mean_sq, mean_sq.shape[1] * mean_sq.shape[2]
    return mean_sq, mean_sq.numel()

def compute_mse(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute normalized MSE in dB."""
    diff = vec1.flatten() - vec2.flatten()
    mse_val = np.mean(diff ** 2)
    denom = 0.5 * (np.mean(vec1.flatten()**2) + np.mean(vec2.flatten()**2))
    nmse_val = mse_val / (denom + 1e-12)
    nmse_dB = 10 * np.log10(nmse_val + 1e-12)
    return nmse_dB

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Pearson correlation."""
    return np.nan if a.size < 2 or b.size < 2 else pearsonr(a.ravel(), b.ravel())[0]

def _layer_sort_key(name: str) -> Tuple[int, int]:
    """Sort key for layer names."""
    if name == "input_after_preproc":
        return (-1, -1)
    if name == "conv1":
        return (0, 999)
    m = re.match(r"layer(\d+)(?:_layer_(\d+))?", name)
    if not m:
        return (9999, 9999)
    base = int(m.group(1))
    sub = int(m.group(2)) if m.group(2) else 999
    return base, sub

def process_pickle_file(filepath: Path) -> List[Dict[str, any]]:
    """
    Process a single pickle file and extract R² and MSE metrics for all layers.
    
    Returns a list of dictionaries with metrics for each layer.
    """
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
    except Exception as e:
        logging.warning(f"Failed to load {filepath}: {e}")
        return []
    
    # Parse filename to get metadata
    filename_info = parse_filename(filepath.name)
    
    results = []
    
    # Get original and synthesized outputs
    orig_outputs = data.get("all_outputs_orig", {})
    synth_outputs = data.get("all_outputs_out_dict", {})
    
    # Process each layer
    for layer in METAMER_LAYERS:
        if layer not in orig_outputs or layer not in synth_outputs:
            continue
            
        orig_acts = orig_outputs[layer]
        synth_acts = synth_outputs[layer]
        
        # Compute squared mean activations
        orig_sq, _ = squared_mean_activations(layer, orig_acts)
        synth_sq, _ = squared_mean_activations(layer, synth_acts)
        
        if orig_sq is not None and synth_sq is not None:
            # Convert to numpy
            orig_np = orig_sq.cpu().numpy()
            synth_np = synth_sq.cpu().numpy()
            
            # Compute correlations
            r_value = pearson(orig_np, synth_np)
            r_squared = r_value ** 2 if not np.isnan(r_value) else np.nan
            
            # Compute MSE
            mse_dB = compute_mse(orig_np, synth_np)
            
            # Store results
            result = {
                'layer': layer,
                'r_squared': r_squared,
                'mse_dB': mse_dB,
                **filename_info
            }
            results.append(result)
    
    return results

def create_heatmaps(pickle_dir: Path, output_dir: Path) -> None:
    """
    Create various heatmaps from the processed pickle files.
    """
    # Find all pickle files
    pickle_files = list(pickle_dir.glob("*.pckl"))
    logging.info(f"Found {len(pickle_files)} pickle files")
    
    if not pickle_files:
        logging.error("No pickle files found!")
        return
    
    # Process all files
    all_results = []
    for filepath in pickle_files:
        results = process_pickle_file(filepath)
        all_results.extend(results)
    
    if not all_results:
        logging.error("No valid results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    logging.info(f"Processed {len(df)} layer results")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. R² Heatmap: Layers vs Sound IDs
    create_layer_sound_heatmap(df, output_dir, metric='r_squared', title_suffix='R²')
    
    # 2. MSE Heatmap: Layers vs Sound IDs  
    create_layer_sound_heatmap(df, output_dir, metric='mse_dB', title_suffix='MSE (dB)')
    
    # 3. R² Heatmap: Layers vs Runs
    create_layer_run_heatmap(df, output_dir, metric='r_squared', title_suffix='R²')
    
    # 4. MSE Heatmap: Layers vs Runs
    create_layer_run_heatmap(df, output_dir, metric='mse_dB', title_suffix='MSE (dB)')
    
    # 5. Model comparison heatmaps
    create_model_comparison_heatmaps(df, output_dir)
    
    # 6. Random seed comparison heatmaps
    create_seed_comparison_heatmaps(df, output_dir)
    
    # Save the processed data
    df.to_csv(output_dir / "processed_metamer_data.csv", index=False)
    logging.info(f"Saved processed data to {output_dir / 'processed_metamer_data.csv'}")

def create_layer_sound_heatmap(df: pd.DataFrame, output_dir: Path, metric: str, title_suffix: str) -> None:
    """Create heatmap of layers vs sound IDs."""
    if df.empty:
        return
        
    # Pivot the data
    pivot_data = df.pivot_table(
        values=metric, 
        index='layer', 
        columns='sound_id', 
        aggfunc='mean'
    )
    
    # Sort layers
    pivot_data = pivot_data.reindex(sorted(pivot_data.index, key=_layer_sort_key))
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    if metric == 'r_squared':
        vmin, vmax = 0.8, 1.0
        cmap = 'coolwarm'
    else:  # mse_dB
        vmin, vmax = -40, -20
        cmap = 'viridis'
    
    sns.heatmap(
        pivot_data, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        linewidths=0.5, 
        annot=True, 
        fmt=".3f", 
        annot_kws={"size": 8}
    )
    
    plt.xlabel("Sound ID")
    plt.ylabel("Layer Name")
    plt.title(f"Layer vs Sound ID - {title_suffix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"layer_sound_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved layer vs sound heatmap: {metric}")

def create_layer_run_heatmap(df: pd.DataFrame, output_dir: Path, metric: str, title_suffix: str) -> None:
    """Create heatmap of layers vs run numbers."""
    if df.empty:
        return
        
    # Pivot the data
    pivot_data = df.pivot_table(
        values=metric, 
        index='layer', 
        columns='run_number', 
        aggfunc='mean'
    )
    
    # Sort layers
    pivot_data = pivot_data.reindex(sorted(pivot_data.index, key=_layer_sort_key))
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    if metric == 'r_squared':
        vmin, vmax = 0.8, 1.0
        cmap = 'coolwarm'
    else:  # mse_dB
        vmin, vmax = -40, -20
        cmap = 'viridis'
    
    sns.heatmap(
        pivot_data, 
        cmap=cmap, 
        vmin=vmin, 
        vmax=vmax, 
        linewidths=0.5, 
        annot=True, 
        fmt=".3f", 
        annot_kws={"size": 8}
    )
    
    plt.xlabel("Run Number")
    plt.ylabel("Layer Name")
    plt.title(f"Layer vs Run Number - {title_suffix}")
    plt.tight_layout()
    plt.savefig(output_dir / f"layer_run_{metric}.png", dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved layer vs run heatmap: {metric}")

def create_model_comparison_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmaps comparing robust vs standard models."""
    if df.empty:
        return
    
    # Filter for files that have both robust and standard versions
    sound_ids_with_both = df.groupby('sound_id')['model_type'].nunique()
    sound_ids_with_both = sound_ids_with_both[sound_ids_with_both >= 2].index
    
    if len(sound_ids_with_both) == 0:
        logging.info("No sound IDs with both robust and standard models found")
        return
    
    filtered_df = df[df['sound_id'].isin(sound_ids_with_both)]
    
    # Create comparison heatmaps
    for metric in ['r_squared', 'mse_dB']:
        pivot_data = filtered_df.pivot_table(
            values=metric,
            index='layer',
            columns=['sound_id', 'model_type'],
            aggfunc='mean'
        )
        
        # Sort layers
        pivot_data = pivot_data.reindex(sorted(pivot_data.index, key=_layer_sort_key))
        
        plt.figure(figsize=(16, 8))
        
        if metric == 'r_squared':
            vmin, vmax = 0.8, 1.0
            cmap = 'coolwarm'
        else:  # mse_dB
            vmin, vmax = -40, -20
            cmap = 'viridis'
        
        sns.heatmap(
            pivot_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 6}
        )
        
        plt.xlabel("Sound ID - Model Type")
        plt.ylabel("Layer Name")
        plt.title(f"Robust vs Standard Model Comparison - {metric.upper()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved model comparison heatmap: {metric}")

def create_seed_comparison_heatmaps(df: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmaps comparing different random seeds."""
    if df.empty:
        return
    
    # Get sound IDs with multiple seeds
    seed_counts = df.groupby('sound_id')['random_seed'].nunique()
    sound_ids_with_multiple_seeds = seed_counts[seed_counts >= 2].index
    
    if len(sound_ids_with_multiple_seeds) == 0:
        logging.info("No sound IDs with multiple random seeds found")
        return
    
    filtered_df = df[df['sound_id'].isin(sound_ids_with_multiple_seeds)]
    
    # Create comparison heatmaps
    for metric in ['r_squared', 'mse_dB']:
        pivot_data = filtered_df.pivot_table(
            values=metric,
            index='layer',
            columns=['sound_id', 'random_seed'],
            aggfunc='mean'
        )
        
        # Sort layers
        pivot_data = pivot_data.reindex(sorted(pivot_data.index, key=_layer_sort_key))
        
        plt.figure(figsize=(16, 8))
        
        if metric == 'r_squared':
            vmin, vmax = 0.8, 1.0
            cmap = 'coolwarm'
        else:  # mse_dB
            vmin, vmax = -40, -20
            cmap = 'viridis'
        
        sns.heatmap(
            pivot_data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0.5,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 6}
        )
        
        plt.xlabel("Sound ID - Random Seed")
        plt.ylabel("Layer Name")
        plt.title(f"Random Seed Comparison - {metric.upper()}")
        plt.tight_layout()
        plt.savefig(output_dir / f"seed_comparison_{metric}.png", dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved seed comparison heatmap: {metric}")

def main():
    parser = argparse.ArgumentParser(description="Create heatmaps from merged metamer pickle files")
    parser.add_argument("--pickle_dir", type=Path, default="merged_metamers",
                       help="Directory containing merged pickle files")
    parser.add_argument("--output_dir", type=Path, default="heatmaps",
                       help="Output directory for heatmaps")
    
    args = parser.parse_args()
    
    # Use absolute paths
    pickle_dir = args.pickle_dir.resolve()
    output_dir = args.output_dir.resolve()
    
    if not pickle_dir.exists():
        logging.error(f"Pickle directory not found: {pickle_dir}")
        return
    
    logging.info(f"Processing pickle files from: {pickle_dir}")
    logging.info(f"Output directory: {output_dir}")
    
    create_heatmaps(pickle_dir, output_dir)
    logging.info("Heatmap generation complete!")

if __name__ == "__main__":
    main() 