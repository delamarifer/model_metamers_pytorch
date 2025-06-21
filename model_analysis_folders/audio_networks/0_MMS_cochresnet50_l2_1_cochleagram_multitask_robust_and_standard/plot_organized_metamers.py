#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""plot_organized_metamers.py

Processes organized metamer pickles to generate plots and metrics.
The script expects pickles in the format:
metamer_{loss_function}_RS{random_seed}_I{iterations}_N{num_rep_iter}_{model_type}_S{sound_id}_{sound_name}_subclip{subclip_idx}.pckl

Example
~~~~~~~
python plot_organized_metamers.py /path/to/metamers_organized /path/to/output
"""
from __future__ import annotations

import argparse
import logging
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import datetime
from multiprocessing import Pool, cpu_count
from itertools import groupby
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.io import wavfile
from scipy.signal import stft
from scipy.stats import pearsonr, spearmanr

###############################################################################
# Configuration & constants
###############################################################################

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stdout)

# Define the layer order for sorting
METAMER_LAYERS = [
    'input_after_preproc',
    'conv1',
    'bn1',
    'conv1_relu1',
    'maxpool1',
    'layer1',
    'layer2',
    'layer3',
    'layer4',
    'avgpool',
    'final/signal/word_int',
    'final/signal/speaker_int',
    'final/noise/labels_binary_via_int',
]

SAMPLE_RATE: int = 16000

###############################################################################
# Utility helpers
###############################################################################

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def parse_metamer_filename(filename: str) -> Dict[str, str]:
    """Parse metamer filename to extract metadata."""
    pattern = r"metamer_(?P<loss_function>.*?)_RS(?P<random_seed>\d+)_I(?P<iterations>\d+)_N(?P<num_rep_iter>\d+)_(?P<model_type>.*?)_S(?P<sound_id>\d+)_(?P<sound_name>.*?)_subclip(?P<subclip_idx>\d+)\.pckl"
    match = re.match(pattern, filename)
    if not match:
        raise ValueError(f"Could not parse filename: {filename}")
    return match.groupdict()

def compute_cochleagram(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, *, nperseg: int = 512, noverlap: int = 256) -> np.ndarray:
    _, _, zxx = stft(audio, fs=sample_rate, nperseg=nperseg, noverlap=noverlap)
    return np.abs(zxx)

def save_audio(audio: np.ndarray, filepath: Path, sample_rate: int = SAMPLE_RATE) -> None:
    ensure_dir(filepath.parent)
    wavfile.write(filepath.as_posix(), sample_rate, audio)

def squared_mean_activations(layer: str, acts: torch.Tensor, *, loss_type: str) -> Tuple[Optional[torch.Tensor], int]:
    if acts is None or isinstance(acts, dict):
        return None, 0
    if loss_type == "inversion_loss_layer" or "final" in layer:
        if len(acts.shape) >= 3:
            return acts, acts.shape[1] * acts.shape[2]
        return acts, acts.numel()
    mean_sq = torch.mean(acts ** 2, dim=-1)
    if len(mean_sq.shape) >= 3:
        return mean_sq, mean_sq.shape[1] * mean_sq.shape[2]
    return mean_sq, mean_sq.numel()

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    return np.nan if a.size < 2 or b.size < 2 else pearsonr(a.ravel(), b.ravel())[0]

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return np.nan if a.size < 2 or b.size < 2 else spearmanr(a.ravel(), b.ravel())[0]

def compute_mse(vec1: np.ndarray, vec2: np.ndarray) -> float:
    diff = vec1.flatten() - vec2.flatten()
    mse_val = np.mean(diff ** 2)
    denom = 0.5 * (np.mean(vec1.flatten()**2) + np.mean(vec2.flatten()**2))
    nmse_val = mse_val / (denom + 1e-12)
    nmse_dB = 10 * np.log10(nmse_val + 1e-12)
    return nmse_dB

def compute_and_plot_correlation_histogram(layer_name: str, ax, cochleagram_seed_1: np.ndarray, cochleagram_seed_2: np.ndarray):
    """Compute and plot correlation histogram between cochleagrams from two different random seeds.
    
    Args:
        layer_name: Name of the layer being analyzed
        ax: Matplotlib axis to plot on
        cochleagram_seed_1: Cochleagram from first random seed
        cochleagram_seed_2: Cochleagram from second random seed
    """
    num_freq_bins = cochleagram_seed_1.shape[0]
    pearson_correlation_coefficients = []
    spearman_correlation_coefficients = []
    for freq_bin in range(num_freq_bins):
        time_series_seed_1 = cochleagram_seed_1[freq_bin, :]
        time_series_seed_2 = cochleagram_seed_2[freq_bin, :]
        if len(time_series_seed_1) >= 2 and len(time_series_seed_2) >= 2:
            pearson_corr, _ = pearsonr(time_series_seed_1, time_series_seed_2)
            spearman_corr, _ = spearmanr(time_series_seed_1, time_series_seed_2)
        else:
            pearson_corr = np.nan
            spearman_corr = np.nan
        pearson_correlation_coefficients.append(pearson_corr)
        spearman_correlation_coefficients.append(spearman_corr)
    ax.hist(pearson_correlation_coefficients, bins=20, alpha=0.5, color='b', label='Pearson')
    ax.hist(spearman_correlation_coefficients, bins=20, alpha=0.5, color='r', label='Spearman')
    ax.set_xlim([-1, 1])
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Correlation of {layer_name} Cochleagrams\nBetween Two Random Seeds', fontsize=8)
    ax.grid(True)
    ax.legend()

###############################################################################
# Core processing class
###############################################################################

class OrganizedMetamerProcessor:
    def __init__(self, *, base_path: Path, output_root: Path) -> None:
        self.base = base_path
        self.out = output_root
        self.r2_rows: List[Dict[str, str | float | int]] = []
        self.mse_rows: List[Dict[str, str | float | int]] = []

    def run(self) -> None:
        """Process all organized metamer files in parallel by sound ID."""
        pickles = self._discover_pickles()
        pickles_by_sound = self._group_by_sound_id(pickles)
        logging.info("Processing %d unique sound IDs", len(pickles_by_sound))
        with Pool(min(cpu_count(), len(pickles_by_sound))) as pool:
            pool.map(self._process_sound_group, pickles_by_sound.values())
        self._generate_summaries()

    def _group_by_sound_id(self, pickles: List[str]) -> Dict[str, List[str]]:
        """Group pickles by sound ID, prioritizing STANDARD files."""
        def sort_key(p: str) -> Tuple[bool, str]:
            metadata = parse_metamer_filename(os.path.basename(p))
            # False comes before True in sorting, so STANDARD (False) will come before ROBUST (True)
            is_robust = metadata['model_type'] == 'ROBUST'
            return (is_robust, p)
        
        # Sort pickles to prioritize STANDARD files
        sorted_pickles = sorted(pickles, key=sort_key)
        
        # Group by sound ID
        groups = {}
        for p in sorted_pickles:
            metadata = parse_metamer_filename(os.path.basename(p))
            sound_id = metadata['sound_id']
            if sound_id not in groups:
                groups[sound_id] = []
            groups[sound_id].append(p)
        
        return groups

    def _process_sound_group(self, pickles: List[str]) -> None:
        """Process a group of pickles with the same sound ID."""
        for p in pickles:
            self._process_pickle(p)

    def _discover_pickles(self) -> List[str]:
        """Find all metamer pickle files in the organized directory."""
        self.pickles = []
        for root, _, files in os.walk(self.base):
            for file in files:
                if file.endswith('.pckl'):
                    full_path = os.path.join(root, file)
                    metadata = parse_metamer_filename(file)
                    if metadata and not metadata['model_type'] == 'ROBUST':  # Only process STANDARD files
                        self.pickles.append(full_path)
        
        # Sort pickles to prioritize STANDARD files
        def sort_key(pickle_path):
            metadata = parse_metamer_filename(os.path.basename(pickle_path))
            return (metadata['model_type'] == 'ROBUST', metadata['sound_id'])
        
        self.pickles.sort(key=sort_key)
        logging.info("Found %d STANDARD pickle files", len(self.pickles))
        return self.pickles

    def _process_pickle(self, p: str) -> None:
        """Process a single metamer pickle file."""
        logging.info("Processing %s", p)
        metadata = parse_metamer_filename(os.path.basename(p))
        d = self._load(p)
        
        # Create output directory structure
        out_dir = self._create_output_dirs(metadata)
        
        # Process each layer
        for layer in self._get_layers(d):
            if layer in {"final", "avgpool_cumulative"}:
                continue
            self._process_layer(layer, d, metadata, out_dir)

    def _create_output_dirs(self, metadata: Dict[str, str]) -> Path:
        """Create the output directory structure."""
        base_dir = self.out / f"{metadata['model_type']}_RS{metadata['random_seed']}_I{metadata['iterations']}_N{metadata['num_rep_iter']}"
        sound_dir = base_dir / f"sound_{metadata['sound_id']}_{metadata['sound_name']}"
        subclip_dir = sound_dir / f"subclip_{metadata['subclip_idx']}"
        metrics_dir = subclip_dir / "metrics"
        summary_dir = sound_dir / "summary"
        
        for dir_path in [base_dir, sound_dir, subclip_dir, metrics_dir, summary_dir]:
            ensure_dir(dir_path)
        
        return subclip_dir

    def _process_layer(self, layer: str, data: dict, metadata: Dict[str, str], out_dir: Path) -> None:
        """Process a single layer and generate all plots."""
        layer_dir = out_dir / f"layer_{layer}"
        ensure_dir(layer_dir)
        
        # Generate base filename with metadata
        base_filename = f"RS{metadata['random_seed']}_I{metadata['iterations']}_N{metadata['num_rep_iter']}_S{metadata['sound_id']}_{metadata['sound_name']}_subclip{metadata['subclip_idx']}"
        
        # Get data
        orig = data["all_outputs_orig"]
        synth = data["all_outputs_out_dict"][layer]
        
        # Generate and save combined plot
        self._generate_combined_plot(layer, orig, synth, data, layer_dir, base_filename)
        
        # Save audio
        synth_audio = data["xadv_dict"][layer].cpu().numpy().ravel()
        audio_path = layer_dir / f"synth_audio_{base_filename}.wav"
        save_audio(synth_audio, audio_path)
        
        # Update metrics
        self._update_metrics(layer, orig, synth, metadata)

    def _generate_combined_plot(self, layer: str, orig: dict, synth: dict, data: dict, out_dir: Path, base_filename: str) -> None:
        """Generate a combined plot with all analyses for a layer."""
        # Create figure with subplots
        fig, axs = plt.subplots(7, 1, figsize=(10, 35))
        fig.suptitle(f"Layer Analysis: {layer}", fontsize=16)
        
        # 1. Cochleagram comparison
        orig_coch = np.squeeze(orig["input_after_preproc"][0].cpu().numpy())
        synth_coch = np.squeeze(synth["input_after_preproc"][0].cpu().numpy())
        axs[0].imshow(synth_coch, origin="lower", aspect="auto")
        axs[0].set_title("Synthetic Cochleagram")
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Frequency")
        
        # 2. Activation scatter
        orig_acts = np.squeeze(orig[layer].cpu().numpy())
        synth_acts = np.squeeze(synth[layer].cpu().numpy())
        axs[1].scatter(orig_acts.ravel(), synth_acts.ravel(), alpha=0.5)
        r = pearson(orig_acts, synth_acts)
        axs[1].set_title(f"Activation Scatter (R² = {r**2:.3f})")
        axs[1].set_xlabel("Original Activations")
        axs[1].set_ylabel("Synthetic Activations")
        
        # 3. Layer correlations
        layer_names = list(data["all_outputs_out_dict"].keys())
        correlations = [pearson(
            np.squeeze(orig[l].cpu().numpy()),
            np.squeeze(synth[l].cpu().numpy())
        ) for l in layer_names]
        axs[2].plot(layer_names, correlations, marker='o')
        axs[2].set_title("Layer-wise Correlations")
        axs[2].set_xlabel("Layer")
        axs[2].set_ylabel("Correlation")
        axs[2].set_xticklabels(layer_names, rotation=45)
        axs[2].grid(True)
        
        # 4. Frequency correlations
        freq_correlations = [pearson(
            orig_coch[f, :],
            synth_coch[f, :]
        ) for f in range(orig_coch.shape[0])]
        axs[3].hist(freq_correlations, bins=20, alpha=0.5)
        axs[3].set_title("Frequency Channel Correlations")
        axs[3].set_xlabel("Correlation")
        axs[3].set_ylabel("Count")
        axs[3].grid(True)
        
        # 5. Original cochleagram
        axs[4].imshow(orig_coch, origin="lower", aspect="auto")
        axs[4].set_title("Original Cochleagram")
        axs[4].set_xlabel("Time")
        axs[4].set_ylabel("Frequency")
        
        # 6. Correlation histogram (if available)
        if "all_outputs_out_dict" in data and len(data["all_outputs_out_dict"]) > 1:
            # Only show correlation histogram if we have two different random seeds
            if "seed2" in data and data["seed2"] is not None:
                compute_and_plot_correlation_histogram(layer, axs[5], orig_coch, synth_coch)
            else:
                axs[5].text(0.5, 0.5, 'No second seed available for correlation', ha='center', va='center')
                axs[5].set_axis_off()
        else:
            axs[5].text(0.5, 0.5, 'No correlation data available', ha='center', va='center')
            axs[5].set_axis_off()
        
        # 7. Loss curve
        if "all_losses_dict" in data and layer in data["all_losses_dict"]:
            iterations = list(data["all_losses_dict"][layer].keys())
            losses = [data["all_losses_dict"][layer][it].item() for it in iterations]
            axs[6].plot(iterations, losses, marker='o')
            axs[6].set_title("Loss Curve")
            axs[6].set_xlabel("Iteration")
            axs[6].set_ylabel("Loss")
            axs[6].grid(True)
        else:
            axs[6].text(0.5, 0.5, 'No loss data available', ha='center', va='center')
            axs[6].set_axis_off()
        
        # Save the combined plot
        plt.tight_layout()
        plt.savefig(out_dir / f"layer_analysis_{base_filename}.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _update_metrics(self, layer: str, orig: dict, synth: dict, metadata: Dict[str, str]) -> None:
        """Update R² and MSE metrics."""
        orig_acts = np.squeeze(orig[layer].cpu().numpy())
        synth_acts = np.squeeze(synth[layer].cpu().numpy())
        
        r2 = pearson(orig_acts, synth_acts) ** 2
        mse = compute_mse(orig_acts, synth_acts)
        
        self.r2_rows.append({
            "LayerName": layer,
            "SoundID": metadata["sound_id"],
            "R^2": r2,
            **metadata
        })
        
        self.mse_rows.append({
            "LayerName": layer,
            "SoundID": metadata["sound_id"],
            "MSE (dB)": mse,
            **metadata
        })

    def _generate_summaries(self) -> None:
        """Generate summary statistics and heatmaps at each level."""
        if not self.r2_rows or not self.mse_rows:
            logging.warning("No metrics collected – skipping summary generation.")
            return
        
        # Convert to DataFrames
        df_r2 = pd.DataFrame(self.r2_rows)
        df_mse = pd.DataFrame(self.mse_rows)
        
        # Generate summaries at each level
        self._generate_level_summaries(df_r2, df_mse, "model_type")
        self._generate_level_summaries(df_r2, df_mse, ["model_type", "random_seed", "iterations", "num_rep_iter"])
        self._generate_level_summaries(df_r2, df_mse, ["model_type", "random_seed", "iterations", "num_rep_iter", "sound_id"])
        self._generate_level_summaries(df_r2, df_mse, ["model_type", "random_seed", "iterations", "num_rep_iter", "sound_id", "subclip_idx"])

    def _generate_level_summaries(self, df_r2: pd.DataFrame, df_mse: pd.DataFrame, group_cols: List[str]) -> None:
        """Generate summary statistics and heatmaps for a specific grouping level."""
        # Group the data
        groups_r2 = df_r2.groupby(group_cols)
        groups_mse = df_mse.groupby(group_cols)
        
        # Generate summaries for each group
        for name, group in groups_r2:
            # Create output directory
            if isinstance(name, tuple):
                out_dir = self.out / "/".join(str(x) for x in name)
            else:
                out_dir = self.out / str(name)
            summary_dir = out_dir / "summary"
            ensure_dir(summary_dir)
            
            # Generate heatmaps
            self._generate_heatmap(group, "R^2", summary_dir / "r2_heatmap.png", vmin=0.9, vmax=1.0)
            self._generate_heatmap(groups_mse.get_group(name), "MSE (dB)", summary_dir / "mse_heatmap.png", vmin=-40, vmax=-20)
            
            # Save metrics
            group.to_csv(summary_dir / "r2_metrics.csv", index=False)
            groups_mse.get_group(name).to_csv(summary_dir / "mse_metrics.csv", index=False)

    def _generate_heatmap(self, df: pd.DataFrame, value_col: str, out_path: Path, vmin: float, vmax: float) -> None:
        """Generate a heatmap from the given DataFrame."""
        heat = df.pivot(index="LayerName", columns="SoundID", values=value_col)
        heat = heat.reindex(sorted(heat.index, key=lambda x: METAMER_LAYERS.index(x) if x in METAMER_LAYERS else 9999))
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(heat, cmap="coolwarm" if value_col == "R^2" else "viridis",
                   vmin=vmin, vmax=vmax, linewidths=0.5, annot=True, fmt=".2f",
                   annot_kws={"size": 8})
        plt.xlabel("Sound ID")
        plt.ylabel("Layer Name")
        plt.title(f"{value_col} Between Synth. & Original Activations")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _load(path: str) -> dict:
        """Load a pickle file."""
        with open(path, "rb") as fh:
            return pickle.load(fh, encoding='bytes')

    @staticmethod
    def _get_layers(data: dict) -> List[str]:
        """Get the list of layers from the data."""
        return sorted(data["all_outputs_out_dict"].keys(), key=lambda x: METAMER_LAYERS.index(x) if x in METAMER_LAYERS else 9999)

###############################################################################
# CLI
###############################################################################

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process organized metamer pickles and generate plots.")
    p.add_argument("base_path", type=Path, help="Path to the organized metamer pickles")
    p.add_argument("output_folder", type=Path, help="Path to save the output plots and metrics")
    return p.parse_args(argv)

def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    processor = OrganizedMetamerProcessor(base_path=args.base_path, output_root=args.output_folder)
    processor.run()

if __name__ == "__main__":
    main() 