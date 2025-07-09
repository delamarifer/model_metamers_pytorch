#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to create heatmaps from merged metamer pickle files.

This script processes the merged pickle files with consistent naming to generate
R^2 and MSE heatmaps across different dimensions (layers, sounds, runs, etc.).

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
import matplotlib.colors as mcolors

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
    run_{run_number}_natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS{seed}_I{iterations}_N8_LR{lr}_DECAY{decay}_{model_type}_SOUNDID_{sound_id}_SUBCLIP{subclip_idx}.pckl
    
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
    
    # Extract model type (any uppercase word/underscore before _SOUNDID)
    model_match = re.search(r'_([A-Z_]+)_SOUNDID', name)
    model_type = model_match.group(1) if model_match else "unknown"
    if model_type == "unknown":
        logging.warning(f"Could not parse model_type from filename: {filename}")
    
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
    
    # Find overlapping layers - synthesized outputs are nested dictionaries
    orig_layers = set(orig_outputs.keys())
    synth_layers = set()
    for layer_name, layer_data in synth_outputs.items():
        if isinstance(layer_data, dict):
            synth_layers.update(layer_data.keys())
        else:
            synth_layers.add(layer_name)
    
    overlapping_layers = sorted(list(orig_layers & synth_layers), key=lambda x: (_layer_sort_key(x), x))
    if not overlapping_layers:
        logging.warning(f"No overlapping layers found in {filepath.name}.\n  orig_layers: {list(orig_layers)}\n  synth_layers: {list(synth_layers)}")
        return []
    logging.info(f"Processing {filepath.name}: overlapping layers: {overlapping_layers}")
    
    # Process each overlapping layer
    for layer in overlapping_layers:
        orig_acts = orig_outputs[layer]
        
        # Handle nested structure of synthesized outputs
        synth_acts = None
        for layer_name, layer_data in synth_outputs.items():
            if isinstance(layer_data, dict) and layer in layer_data:
                synth_acts = layer_data[layer]
                break
            elif layer_name == layer:
                synth_acts = layer_data
                break
        
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

def group_multiple_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group multiple runs for the same sound ID and model type, adding subscripts.
    
    For example, if sound ID 10 has runs 44923809 and 46929020 for robust model,
    they will become "10a" and "10b" respectively.
    """
    # Create a copy to avoid modifying the original
    df_grouped = df.copy()
    
    # Group by sound_id, model_type, and subclip_idx to identify multiple runs
    grouped = df_grouped.groupby(['sound_id', 'model_type', 'subclip_idx'])
    
    # Add run suffix to sound_id for multiple runs
    def add_run_suffix(group):
        if len(group['run_number'].unique()) > 1:
            # Sort by run number for consistent ordering
            unique_runs = sorted(group['run_number'].unique())
            run_to_suffix = {run: chr(97 + i) for i, run in enumerate(unique_runs)}  # a, b, c, ...
            
            # Create new sound_id with suffix
            group['sound_id_grouped'] = group['sound_id'] + group['run_number'].map(run_to_suffix)
        else:
            # Single run, no suffix needed
            group['sound_id_grouped'] = group['sound_id']
        
        return group
    
    # Apply the grouping
    df_grouped = grouped.apply(add_run_suffix).reset_index(drop=True)
    
    return df_grouped

def create_simplified_heatmaps(df: pd.DataFrame, output_dir: Path, all_x_axis_ids=None, suffix="") -> None:
    """
    Create 4 simplified heatmaps: Standard/Robust models x R²/MSE metrics.
    X-axis: Sound ID, Random Seed, Run Number
    Y-axis: Layers
    """
    # Convert axis components to integers for sorting
    # Extract numeric part from sound_id_grouped (e.g., '18a' -> 18)
    df['sound_id_int'] = df['sound_id_grouped'].str.extract(r'(\d+)').astype(int)
    df['subclip_int'] = df['subclip_idx'].astype(int)
    df['random_seed_int'] = df['random_seed'].astype(int)
    df['run_number_int'] = df['run_number'].astype(int)

    # Sort DataFrame by all axis components numerically
    df = df.sort_values(['sound_id_int', 'subclip_int', 'random_seed_int', 'run_number_int'])

    # Rebuild x_axis_id in the correct order
    df['x_axis_id'] = (
        df['sound_id_grouped'] + '_RS' + df['random_seed'] +
        '_R' + df['run_number'] + '_S' + df['subclip_idx'].astype(str)
    )

    # Set x_axis_id as categorical with the order as it appears in the sorted DataFrame, ensuring uniqueness
    x_axis_order = list(dict.fromkeys(df['x_axis_id'].tolist()))
    df['x_axis_id'] = pd.Categorical(df['x_axis_id'], categories=x_axis_order, ordered=True)
    
    # Automatically detect all unique model types in the data
    unique_model_types = df['model_type'].unique()
    for model_type in unique_model_types:
        for metric in ['r_squared', 'mse_dB']:
            create_model_metric_heatmap(df, output_dir, model_type, metric, all_x_axis_ids, suffix)

def create_model_metric_heatmap(df: pd.DataFrame, output_dir: Path, model_type: str, metric: str, all_x_axis_ids=None, suffix="") -> None:
    """
    Create a heatmap for a specific model type and metric.
    
    Args:
        df: DataFrame with the data
        output_dir: Output directory for the heatmap
        model_type: Type of model (e.g., 'SPECTEMP')
        metric: Metric to plot ('r_squared' or 'mse_dB')
        all_x_axis_ids: List of all expected x-axis IDs for consistent ordering
        suffix: Suffix to add to filename (e.g., "_best_optimized")
    """
    # Filter data for this model type
    model_data = df[df['model_type'] == model_type].copy()
    
    if model_data.empty:
        logging.info(f"No data found for model type: {model_type}")
        return
    
    # Create pivot table
    pivot_data = model_data.pivot_table(
        values=metric,
        index='layer',
        columns='x_axis_id',
        aggfunc='first'  # Take first value if duplicates exist
    )
    
    # Ensure all expected columns are present
    if all_x_axis_ids is not None:
        # Create a new DataFrame with all expected columns
        all_cols_df = pd.DataFrame(index=pivot_data.index)
        
        # Sort columns by sound ID, then seed, then run
        def sort_x_axis(col):
            # Extract sound_id, seed, run from column name like "14a_RS9_R44923809"
            parts = col.split('_')
            sound_id = parts[0]
            seed = int(parts[1].replace('RS', ''))
            run = int(parts[2].replace('R', ''))
            return (sound_id, seed, run)
        
        # Add all expected columns in sorted order
        for col in sorted(all_x_axis_ids, key=sort_x_axis):
            if col in pivot_data.columns:
                all_cols_df[col] = pivot_data[col]
            else:
                all_cols_df[col] = 999  # Fill missing data with 999
        
        pivot_data = all_cols_df
    
    # Debug: print actual columns in pivot_data before plotting
    print(f"[DEBUG] Columns in pivot_data before plotting: {list(pivot_data.columns)[:50]} ... total: {len(pivot_data.columns)}")
    
    # Calculate dynamic figure width based on number of columns
    num_columns = len(pivot_data.columns)
    # Use 0.1 inches per column as a reasonable spacing, with minimum 40 inches
    dynamic_width = max(40, num_columns * 0.1)
    print(f"[DEBUG] Using figure width: {dynamic_width} inches for {num_columns} columns")
    
    # Debug: Check actual data values for some specific SOUND IDs
    print(f"\n[DEBUG] Sample data values:")
    for sound_id in [0, 1, 2, 3, 4, 5, 6, 7]:
        for subclip in [0, 1, 2]:
            col_name = f"{sound_id}_RS0_R1_S{subclip}"
            if col_name in pivot_data.columns:
                value = pivot_data[col_name].iloc[0] if len(pivot_data) > 0 else "N/A"
                print(f"  {col_name}: {value}")
            else:
                print(f"  {col_name}: COLUMN MISSING")
    
    # Debug: Count how many values are 999 vs actual data
    if len(pivot_data) > 0:
        first_row = pivot_data.iloc[0]
        missing_count = (first_row == 999).sum()
        data_count = len(first_row) - missing_count
        print(f"\n[DEBUG] Data summary: {data_count} actual values, {missing_count} missing (999) values")
        print(f"[DEBUG] Sample actual values (non-999): {first_row[first_row != 999].head().tolist()}")
        print(f"[DEBUG] Sample missing values (999): {first_row[first_row == 999].head().tolist()}")

    # Create heatmap
    plt.figure(figsize=(dynamic_width, 10))  # Dynamic width based on number of columns

    if metric == 'r_squared':
        # Create a masked array to exclude 999 values from colorbar
        data_for_plotting = pivot_data.copy()
        mask_999 = data_for_plotting == 999
        data_for_plotting_masked = np.ma.masked_where(mask_999, data_for_plotting)
        
        # Create heatmap with custom range for actual data
        sns.heatmap(data_for_plotting_masked, annot=False, cmap='viridis', 
                   vmin=0.9, vmax=1.0, cbar_kws={'label': 'R²'})
        
        # Overlay 999 values in red
        if mask_999.any().any():
            sns.heatmap(pivot_data, mask=~mask_999, annot=False, cmap='Reds', 
                       cbar=False, alpha=0.7)
    else:  # mse_dB
        # Create a masked array to exclude 999 values from colorbar
        data_for_plotting = pivot_data.copy()
        mask_999 = data_for_plotting == 999
        data_for_plotting_masked = np.ma.masked_where(mask_999, data_for_plotting)
        
        # Create heatmap with custom range for actual data (-20 to -40)
        sns.heatmap(data_for_plotting_masked, annot=False, cmap='viridis_r', 
                   vmin=-40, vmax=-20, cbar_kws={'label': 'MSE (dB)'})
        
        # Overlay 999 values in red
        if mask_999.any().any():
            sns.heatmap(pivot_data, mask=~mask_999, annot=False, cmap='Reds', 
                       cbar=False, alpha=0.7)
    
    plt.xlabel("Sound ID, Random Seed, Run Number")
    plt.ylabel("Layer")
    plt.title(f"{model_type} {metric.replace('_', ' ').upper()}{suffix}")
    
    # Save the plot
    output_file = output_dir / f"{model_type}_{metric}{suffix}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saved {model_type} {metric} heatmap{suffix}")


def select_best_runs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the best run for each unique combination of soundID, randomSeed, and subclip.
    Best is defined as highest R² and lowest MSE across all layers for that run.
    
    Args:
        df: DataFrame with all runs
    
    Returns:
        DataFrame with only the best runs
    """
    best_runs = []
    
    # Group by the unique combination (model_type, sound_id, random_seed, subclip_idx)
    for (model_type, sound_id, random_seed, subclip_idx), group in df.groupby(['model_type', 'sound_id', 'random_seed', 'subclip_idx']):
        if len(group) == 1:
            # Only one run, keep it
            best_runs.append(group.iloc[0])
        else:
            # Multiple runs, select the best one
            # First, calculate average performance across all layers for each run
            run_performance = []
            for run_number, run_group in group.groupby('run_number'):
                avg_r2 = run_group['r_squared'].mean()
                avg_mse = run_group['mse_dB'].mean()
                run_performance.append({
                    'run_number': run_number,
                    'avg_r2': avg_r2,
                    'avg_mse': avg_mse,
                    'run_group': run_group
                })
            
            # Find the run with highest average R²
            best_r2_run = max(run_performance, key=lambda x: x['avg_r2'])
            
            # Among runs with similar R² (within 0.001), select the one with lowest average MSE
            r2_threshold = best_r2_run['avg_r2'] - 0.001
            similar_r2_runs = [r for r in run_performance if r['avg_r2'] >= r2_threshold]
            
            if len(similar_r2_runs) == 1:
                best_run_group = best_r2_run['run_group']
            else:
                # Multiple runs with similar R², select the one with lowest average MSE
                best_mse_run = min(similar_r2_runs, key=lambda x: x['avg_mse'])
                best_run_group = best_mse_run['run_group']
            
            # Add all rows from the best run group
            best_runs.extend(best_run_group.to_dict('records'))
    
    return pd.DataFrame(best_runs)


def copy_best_files(df: pd.DataFrame, pickle_dir: Path, output_dir: Path) -> None:
    """
    Copy the best-optimized files to a new directory.
    
    Args:
        df: DataFrame with best runs (should have 'filename' column)
        pickle_dir: Source directory with pickle files
        output_dir: Output directory for best files
    """
    output_dir.mkdir(exist_ok=True)
    
    copied_count = 0
    for _, row in df.iterrows():
        source_file = pickle_dir / row['filename']
        dest_file = output_dir / row['filename']
        
        if source_file.exists():
            import shutil
            shutil.copy2(source_file, dest_file)
            copied_count += 1
        else:
            logging.warning(f"Source file not found: {source_file}")
    
    logging.info(f"Copied {copied_count} best-optimized files to {output_dir}")


def save_missing_tuples(df: pd.DataFrame, all_x_axis_ids: list, output_dir: Path) -> None:
    """
    Save missing tuples (combinations that have 999 values) to a CSV file for later rerunning.
    
    Args:
        df: DataFrame with all runs
        all_x_axis_ids: List of all expected x-axis IDs
        output_dir: Output directory for the missing tuples file
    """
    # Get all unique combinations from the data
    existing_combinations = set()
    for _, row in df.iterrows():
        combination = (row['model_type'], row['sound_id'], row['random_seed'], row['subclip_idx'])
        existing_combinations.add(combination)
    
    # Parse all_x_axis_ids to get expected combinations
    expected_combinations = set()
    for x_axis_id in all_x_axis_ids:
        # Parse x_axis_id like "0_RS0_R1_S0" to extract components
        parts = x_axis_id.split('_')
        sound_id = parts[0]
        random_seed = parts[1].replace('RS', '')
        subclip_idx = parts[3].replace('S', '')
        # For now, assume all are SPECTEMP model type
        model_type = 'SPECTEMP'
        expected_combinations.add((model_type, sound_id, random_seed, subclip_idx))
    
    # Find missing combinations
    missing_combinations = expected_combinations - existing_combinations
    
    # Convert to DataFrame for saving
    missing_data = []
    for model_type, sound_id, random_seed, subclip_idx in missing_combinations:
        missing_data.append({
            'model_type': model_type,
            'sound_id': sound_id,
            'random_seed': random_seed,
            'subclip_idx': subclip_idx,
            'x_axis_id': f"{sound_id}_RS{random_seed}_R1_S{subclip_idx}"
        })
    
    if missing_data:
        missing_df = pd.DataFrame(missing_data)
        missing_file = output_dir / "missing_tuples_to_rerun.csv"
        missing_df.to_csv(missing_file, index=False)
        logging.info(f"Saved {len(missing_data)} missing tuples to {missing_file}")
        
        # Also save a summary
        summary_file = output_dir / "missing_tuples_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Missing tuples summary:\n")
            f.write(f"Total missing combinations: {len(missing_data)}\n")
            f.write(f"Missing sound IDs: {sorted(set(missing_df['sound_id']))}\n")
            f.write(f"Missing subclips: {sorted(set(missing_df['subclip_idx']))}\n")
            f.write(f"Missing random seeds: {sorted(set(missing_df['random_seed']))}\n")
    else:
        logging.info("No missing tuples found - all expected combinations are present")


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
    
    # Group multiple runs for the same sound ID and model type
    df = group_multiple_runs(df)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build all possible x_axis_id combinations for full grid (to show gray for missing)
    sound_ids = [str(i) for i in range(42)]
    subclips = [str(i) for i in range(3)]
    random_seeds = sorted(df['random_seed'].unique(), key=lambda x: int(x))
    run_numbers = sorted(df['run_number'].unique(), key=lambda x: int(x))

    all_x_axis_ids = []
    for sid in sound_ids:
        for subclip in subclips:
            for seed in random_seeds:
                for run in run_numbers:
                    all_x_axis_ids.append(f"{sid}_RS{seed}_R{run}_S{subclip}")

    # Rebuild x_axis_id in the correct order for all rows
    df['x_axis_id'] = (
        df['sound_id_grouped'] + '_RS' + df['random_seed'] +
        '_R' + df['run_number'] + '_S' + df['subclip_idx'].astype(str)
    )

    # Set x_axis_id as categorical with the full grid order
    df['x_axis_id'] = pd.Categorical(df['x_axis_id'], categories=all_x_axis_ids, ordered=True)

    # Create simplified heatmaps: 4 total (2 models x 2 metrics) - ALL RUNS
    create_simplified_heatmaps(df, output_dir, all_x_axis_ids=all_x_axis_ids)
    
    # Save the processed data for all runs
    df.to_csv(output_dir / "processed_metamer_data.csv", index=False)
    logging.info(f"Saved processed data to {output_dir / 'processed_metamer_data.csv'}")
    
    # Now create best-optimized version
    logging.info("Creating best-optimized version...")
    
    # Select best runs for each unique combination
    df_best = select_best_runs(df)
    logging.info(f"Selected {len(df_best)} best runs out of {len(df)} total runs")
    
    # Create best-optimized heatmaps
    create_simplified_heatmaps(df_best, output_dir, all_x_axis_ids=all_x_axis_ids, suffix="_best_optimized")
    
    # Save the best-optimized data
    df_best.to_csv(output_dir / "processed_metamer_data_best_optimized.csv", index=False)
    logging.info(f"Saved best-optimized data to {output_dir / 'processed_metamer_data_best_optimized.csv'}")
    
    # Copy best files to new directory
    better_optimized_dir = pickle_dir.parent / "merged_metamers_betteroptimized"
    copy_best_files(df_best, pickle_dir, better_optimized_dir)
    
    # Save missing tuples for later rerunning
    save_missing_tuples(df, all_x_axis_ids, output_dir)

def create_layer_sound_heatmap(df: pd.DataFrame, output_dir: Path, metric: str, title_suffix: str) -> None:
    """Create heatmap of layers vs sound IDs."""
    if df.empty:
        return
        
    # Pivot the data using grouped sound IDs
    pivot_data = df.pivot_table(
        values=metric, 
        index='layer', 
        columns='sound_id_grouped', 
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
        annot=False  # Remove text annotations
    )
    
    plt.xlabel("Sound ID (with run suffixes)")
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
        annot=False  # Remove text annotations
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
    sound_ids_with_both = df.groupby('sound_id_grouped')['model_type'].nunique()
    sound_ids_with_both = sound_ids_with_both[sound_ids_with_both >= 2].index
    
    if len(sound_ids_with_both) == 0:
        logging.info("No sound IDs with both robust and standard models found")
        return
    
    filtered_df = df[df['sound_id_grouped'].isin(sound_ids_with_both)]
    
    # Create comparison heatmaps
    for metric in ['r_squared', 'mse_dB']:
        pivot_data = filtered_df.pivot_table(
            values=metric,
            index='layer',
            columns=['sound_id_grouped', 'model_type'],
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
            annot=False  # Remove text annotations
        )
        
        plt.xlabel("Sound ID (with run suffixes) - Model Type")
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
    seed_counts = df.groupby('sound_id_grouped')['random_seed'].nunique()
    sound_ids_with_multiple_seeds = seed_counts[seed_counts >= 2].index
    
    if len(sound_ids_with_multiple_seeds) == 0:
        logging.info("No sound IDs with multiple random seeds found")
        return
    
    filtered_df = df[df['sound_id_grouped'].isin(sound_ids_with_multiple_seeds)]
    
    # Create comparison heatmaps
    for metric in ['r_squared', 'mse_dB']:
        pivot_data = filtered_df.pivot_table(
            values=metric,
            index='layer',
            columns=['sound_id_grouped', 'random_seed'],
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
            annot=False  # Remove text annotations
        )
        
        plt.xlabel("Sound ID (with run suffixes) - Random Seed")
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
    import argparse
    parser = argparse.ArgumentParser(description="Create metamer heatmaps (compatible with spectemp_filters and cochresnet)")
    parser.add_argument('--pickle_dir', type=str, default='merged_metamers', help='Directory containing metamer pickle files')
    parser.add_argument('--output_dir', type=str, default='heatmaps', help='Directory to save heatmaps')
    args = parser.parse_args()
    from pathlib import Path
    create_heatmaps(Path(args.pickle_dir), Path(args.output_dir)) 