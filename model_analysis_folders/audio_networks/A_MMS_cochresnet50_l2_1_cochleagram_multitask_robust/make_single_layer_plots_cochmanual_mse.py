#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""metamer_processor.py

Processes metamer pickles to generate R² and MSE metrics
plus heatmaps across layers and sounds.

Example
~~~~~~~
python metamer_processor.py /path/to/pickles /path/to/output \
       --rand_seed_1 0 --rand_seed_2 1                        \
       --model_type standard --loss_type square_time_average  \
       --sound_id 8
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

SAMPLE_RATE: int = 20_000

METAMER_LAYERS: List[str] = [
    "input_after_preproc",
    "conv1",
    "layer1",
    "layer1_cumulative",
    "layer2",
    "layer2_cumulative",
    "layer3_layer_1",
    "layer3_layer_2",
    "layer3_layer_3",
    "layer3_layer_4",
    "layer3_layer_5",
    "layer3",
    "layer3_cumulative",
    "layer4_layer_1",
    "layer4_layer_2",
    "layer4",
    "layer4_cumulative",
    "avgpool",
    "avgpool_cumulative",
    "conv1_layer1_cumulative",
    "conv1_layer4_cumulative",
    "layer1_layer3_cumulative",
    "layer1_layer4_cumulative",
    "layer2_layer3_cumulative",
    "layer2_layer4_cumulative",
    "new_layer4_cumulative_layer1",
    "combined_layer1_sub_layer4_layer_1",
    "combined_conv1_sub_layer4_layer_1",
    "combined_conv1_avgpool"
]

###############################################################################
# Utility helpers
###############################################################################

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

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
        return acts, acts.shape[1] * acts.shape[2]
    mean_sq = torch.mean(acts ** 2, dim=-1)
    return mean_sq, mean_sq.shape[1] * mean_sq.shape[2]

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
    ax.set_title(f'Correlation of {layer_name}-MMS Cochleagrams (input_after_preproc)\nfrom two different seeds for Across Frequency Channels', fontsize=8)
    ax.grid(True)
    ax.legend()

def compute_spearman_rho_pair(activations):
    return spearmanr(activations[0].ravel(), activations[1].ravel())[0]

def compute_pearson_corr_pair(activations):
    if len(activations[0]) < 2 or len(activations[1]) < 2:
        return np.nan
    return pearsonr(activations[0].ravel(), activations[1].ravel())[0]

###############################################################################
# Heatmap utilities
###############################################################################

def _layer_sort_key(name: str) -> Tuple[int, int]:
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

def create_heatmaps(csv_dir: Path) -> None:
    # R² heatmap
    csv_files_r2 = list(csv_dir.glob("*_activations_r2.csv"))
    if csv_files_r2:
        df_r2 = pd.concat((pd.read_csv(f) for f in csv_files_r2), ignore_index=True)
        if not df_r2.empty:
            heat_r2 = df_r2.pivot(index="LayerName", columns="SoundID", values="R^2")
            heat_r2 = heat_r2.reindex(sorted(heat_r2.index, key=_layer_sort_key))
            plt.figure(figsize=(10, 6))
            sns.heatmap(heat_r2, cmap="coolwarm", vmin=0.9, vmax=1.0, linewidths=0.5, annot=True, fmt=".2f", annot_kws={"size":8})
            plt.xlabel("Sound ID")
            plt.ylabel("Layer Name")
            plt.title("Correlation (R²) Between Synth. & Original Activations – 3 s")
            plt.savefig(csv_dir / "R2_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()
            logging.info("Saved R² heatmap → %s", csv_dir / "R2_heatmap.png")

    # MSE heatmap
    csv_files_mse = list(csv_dir.glob("*_activations_mse.csv"))
    if csv_files_mse:
        df_mse = pd.concat((pd.read_csv(f) for f in csv_files_mse), ignore_index=True)
        if not df_mse.empty:
            heat_mse = df_mse.pivot(index="LayerName", columns="SoundID", values="MSE (dB)")
            heat_mse = heat_mse.reindex(sorted(heat_mse.index, key=_layer_sort_key))
            plt.figure(figsize=(10, 6))
            sns.heatmap(heat_mse, cmap="viridis", linewidths=0.5, vmin=-40, vmax=-20, annot=True, fmt=".2f", annot_kws={"size":8})
            plt.xlabel("Sound ID")
            plt.ylabel("Layer Name")
            plt.title("NMSE (dB) Between Synth. & Original Activations – 3 s")
            plt.savefig(csv_dir / "MSE_heatmap.png", dpi=300, bbox_inches="tight")
            plt.close()
            logging.info("Saved MSE heatmap → %s", csv_dir / "MSE_heatmap.png")

###############################################################################
# Core processing class
###############################################################################

class MetamerProcessor:
    def __init__(self, *, base_path: Path, output_root: Path, sound_id: Optional[int], rand_seed_1: int, rand_seed_2: Optional[int], model_type: str, loss_type: str) -> None:
        self.base = base_path
        self.out = output_root
        self.sound_id = sound_id
        self.seed1 = rand_seed_1
        self.seed2 = rand_seed_2
        self.model_type = model_type
        self.loss_type = loss_type
        self.r2_rows: List[Dict[str, str | float | int]] = []
        self.mse_rows: List[Dict[str, str | float | int]] = []

    def run(self) -> None:
        for pkl in self._discover_pickles():
            self._process_pickle(pkl)
        self._write_csv()

    def _discover_pickles(self) -> List[Path]:
        target = "all_metamers_pickle.pckl"
        parent_filter = f"{self.sound_id}_SOUND_about" if self.sound_id is not None else None
        results = []
        for root, _, files in os.walk(self.base):
            if target in files:
                p = Path(root) / target
                if parent_filter is None or p.parent.name == parent_filter:
                    results.append(p)
        logging.info("Found %d pickles", len(results))
        return sorted(results)

    def _process_pickle(self, p: Path) -> None:
        logging.info("Processing %s", p)
        d1 = self._load(p)
        d2 = None
        if self.seed2 is not None:
            alt = Path(str(p).replace(f"_RS{self.seed1}", f"_RS{self.seed2}"))
            if alt.exists():
                d2 = self._load(alt)
            else:
                logging.warning("Missing seed-2 pickle %s", alt)
        common_layers = self._common_layers(d1, d2)
        for layer in common_layers:
            if layer in {"final", "avgpool_cumulative"}:
                continue
            self._plot_layer(layer, d1, d2, p)

    @staticmethod
    def _load(path: Path):
        with path.open("rb") as fh:
            return pickle.load(fh)

    def _common_layers(self, d1: dict, d2: Optional[dict]) -> List[str]:
        s1 = set(d1["all_outputs_out_dict"].keys())
        if d2 is None:
            return sorted(s1, key=METAMER_LAYERS.index)
        s2 = set(d2["all_outputs_out_dict"].keys())
        return sorted(s1 & s2, key=METAMER_LAYERS.index)

    def _plot_layer(self, layer: str, d1: dict, d2: Optional[dict], origin: Path) -> None:
        orig = d1["all_outputs_orig"]
        synth_dict = d1["all_outputs_out_dict"][layer]
        synth_coch = np.squeeze(synth_dict["input_after_preproc"][0].cpu().numpy())
        orig_coch = np.squeeze(orig["input_after_preproc"][0].cpu().numpy())
        synth_audio = d1["xadv_dict"][layer].cpu().numpy().ravel()

        # Prepare for extra plots
        # If d2 is provided, get the cochleagram for the same layer from seed2
        synth_coch_seed2 = None
        if d2 is not None and layer in d2["all_outputs_out_dict"]:
            synth_dict_seed2 = d2["all_outputs_out_dict"][layer]
            synth_coch_seed2 = np.squeeze(synth_dict_seed2["input_after_preproc"][0].cpu().numpy())

        # Find common layers for layer-wise correlation plots
        layers_seed_1 = set(d1["all_outputs_out_dict"].keys())
        layers_seed_2 = set(d2["all_outputs_out_dict"].keys()) if d2 else set()
        common_layers = layers_seed_1 & layers_seed_2 if d2 else layers_seed_1
        filtered_layers = [l for l in common_layers if l in METAMER_LAYERS and l not in {"final", "avgpool_cumulative"}]
        common_layers_sorted = sorted(filtered_layers, key=METAMER_LAYERS.index)

        fig, axs = plt.subplots(1, 7, figsize=(42, 5))
        # 0: Synth cochleagram
        axs[0].imshow(synth_coch, origin="lower", aspect="auto")
        axs[0].set_title(f"Synth – {layer}")
        # 1: Scatter activations
        if not isinstance(orig[layer], dict):
            o_sq, _ = squared_mean_activations(layer, orig[layer], loss_type=self.loss_type)
            s_sq, _ = squared_mean_activations(layer, synth_dict[layer], loss_type=self.loss_type)
            if o_sq is not None and s_sq is not None:
                axs[1].scatter(o_sq.cpu().numpy().ravel(), s_sq.cpu().numpy().ravel(), alpha=0.5)
                r = pearson(o_sq.cpu().numpy(), s_sq.cpu().numpy())
                self.r2_rows.append({"SoundID": self.sound_id, "LayerName": layer, "R^2": r ** 2})
                mse = compute_mse(o_sq.cpu().numpy(), s_sq.cpu().numpy())
                self.mse_rows.append({"SoundID": self.sound_id, "LayerName": layer, "MSE (dB)": mse})
        else:
            axs[1].axis("off")
        axs[1].set_title("Scatter: Activations")
        # 2: Original cochleagram
        axs[2].imshow(orig_coch, origin="lower", aspect="auto")
        axs[2].set_title("Original coch")
        # 3: Layer-wise correlation (activations)
        spearman_activations_correlations = []
        pearson_activations_correlations = []
        for other_layer in common_layers_sorted:
            o_sq, _ = squared_mean_activations(other_layer, orig[other_layer], loss_type=self.loss_type)
            s_sq, _ = squared_mean_activations(other_layer, synth_dict[other_layer], loss_type=self.loss_type)
            if o_sq is not None and s_sq is not None:
                o_flat = o_sq.cpu().numpy().ravel()
                s_flat = s_sq.cpu().numpy().ravel()
                spearman_rho = compute_spearman_rho_pair([o_flat, s_flat])
                pearson_rho = compute_pearson_corr_pair([o_flat, s_flat])
            else:
                spearman_rho = pearson_rho = np.nan
            spearman_activations_correlations.append(spearman_rho)
            pearson_activations_correlations.append(pearson_rho)
        axs[3].plot(common_layers_sorted, spearman_activations_correlations, marker='o', label='Spearman')
        axs[3].plot(common_layers_sorted, pearson_activations_correlations, marker='o', label='Pearson', linestyle='--')
        axs[3].set_xlabel('Layers')
        axs[3].set_ylabel('Correlation')
        axs[3].set_title(f'Layer-wise Activation Correlation')
        axs[3].legend()
        axs[3].set_xticklabels(common_layers_sorted, rotation=90)
        axs[3].set_ylim([0, 1])
        # 4: Layer-wise cochleagram correlation
        spearman_cochleagram_correlations = []
        pearson_cochleagram_correlations = []
        for other_layer in common_layers_sorted:
            coch_other = np.squeeze(np.array(d1["all_outputs_out_dict"][other_layer]["input_after_preproc"][0].cpu()))
            time_avg_current = np.mean(synth_coch**2, axis=-1)
            time_avg_other = np.mean(coch_other**2, axis=-1)
            spearman_rho = compute_spearman_rho_pair([time_avg_current, time_avg_other])
            pearson_rho = compute_pearson_corr_pair([time_avg_current, time_avg_other])
            spearman_cochleagram_correlations.append(spearman_rho)
            pearson_cochleagram_correlations.append(pearson_rho)
        axs[4].plot(common_layers_sorted, spearman_cochleagram_correlations, marker='o', label='Spearman')
        axs[4].plot(common_layers_sorted, pearson_cochleagram_correlations, marker='o', label='Pearson', linestyle='--')
        axs[4].set_xlabel('Layers')
        axs[4].set_ylabel('Cochleagram Corr.')
        axs[4].set_title(f'Layer-wise Cochleagram Correlation')
        axs[4].legend()
        axs[4].set_xticklabels(common_layers_sorted, rotation=90)
        axs[4].set_ylim([0, 1])
        # 5: Histogram of freq channel correlations (if seed2 available)
        if synth_coch_seed2 is not None:
            compute_and_plot_correlation_histogram(layer, axs[5], synth_coch, synth_coch_seed2)
        else:
            axs[5].text(0.5, 0.5, 'No seed2', ha='center', va='center', fontsize=12)
            axs[5].set_axis_off()
        # 6: Loss plot
        ax_loss = axs[6]
        loss_data = d1.get("all_losses", {}).get(layer, None)
        if loss_data is not None and isinstance(loss_data, dict) and len(loss_data) > 0:
            iterations = list(loss_data.keys())
            loss_values = [loss_data[it].item() if hasattr(loss_data[it], 'item') else float(loss_data[it]) for it in iterations]
            ax_loss.plot(iterations, loss_values, marker='o', linestyle='-', color='black')
            ax_loss.set_xlabel('Iterations')
            ax_loss.set_ylabel('Loss')
            ax_loss.set_title(f'Loss Curve for {layer}', fontsize=10)
            ax_loss.grid(True)
        else:
            ax_loss.text(0.5, 0.5, 'No loss data', ha='center', va='center', fontsize=12)
            ax_loss.set_axis_off()
        out_dir = self._derive_out_dir(origin, layer)
        ensure_dir(out_dir)
        fig.tight_layout()
        fig.savefig(out_dir / f"combined_activation_{layer}.png", dpi=300)
        plt.close(fig)
        save_audio(synth_audio, out_dir / f"synth_audio_{layer}.wav")

    def _derive_out_dir(self, origin: Path, layer: str) -> Path:
        # Get current timestamp in format YYYYMMDD_HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.out / "RESNET_STANDARD" / origin.parent.name / f"output_rand_seed_{self.seed1}_{layer}_{timestamp}"

    def _write_csv(self) -> None:
        if not self.r2_rows:
            logging.warning("No R² rows collected – skipping CSV export.")
            return
        csv_dir = self.out.with_name(self.out.name + "_R2")
        ensure_dir(csv_dir)
        pd.DataFrame(self.r2_rows).to_csv(csv_dir / f"{self.sound_id}_activations_r2.csv", index=False)
        pd.DataFrame(self.mse_rows).to_csv(csv_dir / f"{self.sound_id}_activations_mse.csv", index=False)
        logging.info("Wrote R² and MSE CSVs → %s", csv_dir)

###############################################################################
# CLI
###############################################################################

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse metamer pickles and produce diagnostic figures + R²/MSE heatmaps.", add_help=False)
    # p.add_argument("base_path", nargs="?", type=Path)
    # p.add_argument("output_folder", nargs="?", type=Path)
    p.add_argument("--sound_id", type=int, default=None)
    p.add_argument("--rand_seed_1", type=int, required=True)
    p.add_argument("--rand_seed_2", type=int, default=None)
    p.add_argument("--model_type", choices=["robust", "standard"], required=True)
    p.add_argument("--loss_type", required=True)
    p.add_argument("--skip_heatmap", action="store_true")
    p.add_argument("--base_path", dest="base_path_kw", type=Path)
    p.add_argument("--output_folder", dest="output_folder_kw", type=Path)

    known, unknown = p.parse_known_args(argv)

    # Reconcile positional vs. keyword arguments
    if known.base_path_kw is not None:
        known.base_path = known.base_path_kw
    if known.output_folder_kw is not None:
        known.output_folder = known.output_folder_kw

    # Validate
    if known.base_path is None or known.output_folder is None:
        p.error("base_path and output_folder are required (either positionally or via --base_path / --output_folder).")

    if "-h" in unknown or "--help" in unknown:
        p.print_help(sys.stderr)
        sys.exit(0)
    if unknown:
        logging.warning("Ignoring unknown CLI arguments: %s", " ".join(unknown))
    return known

def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    processor = MetamerProcessor(
        base_path=args.base_path,
        output_root=args.output_folder,
        sound_id=args.sound_id,
        rand_seed_1=args.rand_seed_1,
        rand_seed_2=args.rand_seed_2,
        model_type=args.model_type,
        loss_type=args.loss_type,
    )
    processor.run()
    if not args.skip_heatmap:
        create_heatmaps(args.output_folder.with_name(args.output_folder.name + "_R2"))

if __name__ == "__main__":
    main()
