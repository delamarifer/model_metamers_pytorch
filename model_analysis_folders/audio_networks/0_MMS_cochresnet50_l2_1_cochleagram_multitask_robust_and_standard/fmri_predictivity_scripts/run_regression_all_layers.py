import os
import sys
import subprocess
import h5py
import time
from run_regression_single_layer import run_regression, get_layer_info

# Settings
MODEL_TYPES = ['robust', 'standard']
FEATURES_DIR = './features'

def estimate_total_time(model_types):
    """Estimate total time for all regressions"""
    total_estimated_minutes = 0
    
    for model_type in model_types:
        try:
            layer_list = get_layer_info(model_type)
            print(f"Model {model_type} has {len(layer_list)} layers:")
            for i, layer in enumerate(layer_list):
                # Rough estimate based on layer position (deeper layers typically have more features)
                if i < 3:
                    estimated_minutes = 1
                elif i < 5:
                    estimated_minutes = 2
                else:
                    estimated_minutes = 3
                total_estimated_minutes += estimated_minutes
                print(f"  Layer {i}: {layer} (est. {estimated_minutes}m)")
        except FileNotFoundError:
            print(f"Warning: No feature file found for {model_type}")
    
    hours = int(total_estimated_minutes // 60)
    minutes = int(total_estimated_minutes % 60)
    return f"{hours}h {minutes}m", total_estimated_minutes

def main():
    print("=== fMRI Regression Analysis for All Layers ===")
    
    # Check if features exist
    missing_features = []
    for model_type in MODEL_TYPES:
        feature_file = os.path.join(FEATURES_DIR, f'natsound_activations_{model_type}.h5')
        if not os.path.exists(feature_file):
            missing_features.append(model_type)
    
    if missing_features:
        print(f"Error: Feature files missing for: {missing_features}")
        print("Please run extract_features_only.py first to generate features.")
        return 1
    
    # Estimate total time
    estimated_time_str, total_minutes = estimate_total_time(MODEL_TYPES)
    print(f"\nEstimated total time for all regressions: {estimated_time_str}")
    
    # Ask for confirmation
    response = input("\nProceed with all regressions? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return 0
    
    # Run regressions for all layers
    start_time = time.time()
    successful_regressions = 0
    total_regressions = 0
    
    for model_type in MODEL_TYPES:
        print(f"\n=== Processing model: {model_type} ===")
        layer_list = get_layer_info(model_type)
        
        for layer_idx in range(len(layer_list)):
            total_regressions += 1
            print(f"\n--- Layer {layer_idx}/{len(layer_list)-1}: {layer_list[layer_idx]} ---")
            
            # Estimate remaining time
            elapsed_minutes = (time.time() - start_time) / 60
            if successful_regressions > 0:
                avg_time_per_regression = elapsed_minutes / successful_regressions
                remaining_regressions = (len(MODEL_TYPES) * len(layer_list)) - total_regressions
                estimated_remaining = remaining_regressions * avg_time_per_regression
                print(f"Progress: {total_regressions}/{(len(MODEL_TYPES) * len(layer_list))} regressions")
                print(f"Elapsed: {elapsed_minutes:.1f}m, Estimated remaining: {estimated_remaining:.1f}m")
            
            # Run regression
            success = run_regression(model_type, layer_idx)
            if success:
                successful_regressions += 1
            else:
                print(f"Warning: Regression failed for {model_type} layer {layer_idx}")
    
    # Summary
    total_time = time.time() - start_time
    print(f"\n=== Regression Analysis Complete ===")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Successful regressions: {successful_regressions}/{total_regressions}")
    print(f"Success rate: {successful_regressions/total_regressions*100:.1f}%")
    
    if successful_regressions == total_regressions:
        print("\nAll regressions completed successfully!")
        print("You can now run plotting with: python plot_fmri_predictivity_all_layers.py")
        return 0
    else:
        print(f"\nWarning: {total_regressions - successful_regressions} regressions failed.")
        return 1

if __name__ == "__main__":
    exit(main()) 