import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

# Add parent directory to path to import build_network
sys.path.append('..')
import build_network

from datetime import datetime

MODEL_TYPES = ['robust', 'standard']
FEATURES_DIR = './features'
REGRESSION_RESULTS_DIR = './regression_results'  # This should match the output dir in regression script

# Create informative plots directory name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
PLOTS_DIR = f'./plots_fmri_predictivity_analysis_{timestamp}'
os.makedirs(PLOTS_DIR, exist_ok=True)
print(f"Plots will be saved to: {PLOTS_DIR}")

# Helper to find the regression result file for a given model/layer
def find_regression_pckl(model_type, layer_name):
    # The regression script saves results in regression_results/<feature_file_basename>/<...>.pckl
    # We'll look for the file in the expected subdirectory
    feature_file_base = f'{model_type}_{layer_name}'
    subdir = os.path.join(REGRESSION_RESULTS_DIR, feature_file_base)
    if not os.path.isdir(subdir):
        return None
    for fname in os.listdir(subdir):
        if fname.endswith('.pckl'):
            return os.path.join(subdir, fname)
    return None

# Main plotting loop
for model_type in MODEL_TYPES:
    print(f'Processing model: {model_type}')
    model, ds, metamer_layers = build_network.main(return_metamer_layers=True, model_type=model_type)
    metamer_layers = [l for l in metamer_layers if not l.startswith('#')]
    median_r2s = []
    sem_r2s = []
    valid_layers = []
    for layer_idx, layer_name in enumerate(metamer_layers):
        pckl_path = find_regression_pckl(model_type, layer_name)
        if pckl_path is None:
            print(f'  [Missing] No regression result for {model_type} {layer_name}')
            continue
        print(f'  Loading: {pckl_path}')
        with open(pckl_path, 'rb') as f:
            info = pickle.load(f)
        # info['r2s'] shape: [voxels, splits]
        r2s = info['r2s']
        # Compute median across voxels, then mean/sem across splits
        # Use the same logic as compute_median_across_predictions in voxel_regression_functions.py
        # But here, just take the median across voxels for each split, then mean/sem across splits
        median_voxel_r2 = np.nanmedian(r2s, 0)  # median across voxels, for each split
        mean_r2 = np.nanmean(median_voxel_r2)
        sem_r2 = np.nanstd(median_voxel_r2) / np.sqrt(len(median_voxel_r2))
        median_r2s.append(mean_r2)
        sem_r2s.append(sem_r2)
        valid_layers.append(layer_name)
    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(valid_layers)), median_r2s, yerr=sem_r2s, capsize=5)
    plt.xticks(range(len(valid_layers)), valid_layers, rotation=45, ha='right')
    plt.ylabel('Median R^2 (fMRI predictivity)')
    plt.xlabel('Layer')
    plt.title(f'fMRI Predictivity by Layer ({model_type})')
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, f'fmri_predictivity_{model_type}.png')
    plt.savefig(plot_path)
    print(f'  Saved plot: {plot_path}')
    plt.close() 