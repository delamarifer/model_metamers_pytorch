#!/usr/bin/env python3
"""
fMRI Predictivity Analysis Workflow

This script provides multiple options for running fMRI predictivity analysis:
1. Extract features only (run once)
2. Run regressions sequentially (all layers)
3. Run regressions in parallel (individual layers)
4. Run everything (features + regressions)

Time estimates and progress tracking are provided for each option.
"""

import os
import sys
import subprocess
import time
import argparse
from run_regression_single_layer import get_layer_info, estimate_total_time

def print_banner():
    print("=" * 60)
    print("fMRI Predictivity Analysis Workflow")
    print("=" * 60)
    print("Available layers for each model:")
    
    MODEL_TYPES = ['robust', 'standard']
    for model_type in MODEL_TYPES:
        try:
            layer_list = get_layer_info(model_type)
            print(f"\n{model_type.upper()} model ({len(layer_list)} layers):")
            for i, layer in enumerate(layer_list):
                print(f"  {i}: {layer}")
        except FileNotFoundError:
            print(f"\n{model_type.upper()} model: No feature file found")
    
    print("\n" + "=" * 60)

def estimate_feature_extraction_time():
    """Estimate time for feature extraction"""
    # 165 sounds * 7 layers * 2 models * 0.1 seconds per operation
    total_operations = 165 * 7 * 2 * 0.1
    hours = int(total_operations // 3600)
    minutes = int((total_operations % 3600) // 60)
    return f"{hours}h {minutes}m"

def check_features_exist():
    """Check if feature files exist"""
    MODEL_TYPES = ['robust', 'standard']
    missing = []
    for model_type in MODEL_TYPES:
        feature_file = f'./features/natsound_activations_{model_type}.h5'
        if not os.path.exists(feature_file):
            missing.append(model_type)
    return missing

def run_feature_extraction():
    """Run feature extraction"""
    print("=== Running Feature Extraction ===")
    estimated_time = estimate_feature_extraction_time()
    print(f"Estimated time: {estimated_time}")
    
    response = input("Proceed with feature extraction? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return False
    
    start_time = time.time()
    result = subprocess.run(['python', 'extract_features_only.py'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        total_time = time.time() - start_time
        print(f"Feature extraction completed in {total_time/60:.1f} minutes")
        print("Output:", result.stdout)
        return True
    else:
        print("Feature extraction failed!")
        print("Error:", result.stderr)
        return False

def run_sequential_regressions():
    """Run regressions sequentially"""
    print("=== Running Sequential Regressions ===")
    
    missing_features = check_features_exist()
    if missing_features:
        print(f"Error: Feature files missing for: {missing_features}")
        print("Please run feature extraction first.")
        return False
    
    estimated_time_str, _ = estimate_total_time(['robust', 'standard'])
    print(f"Estimated time: {estimated_time_str}")
    
    response = input("Proceed with sequential regressions? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return False
    
    result = subprocess.run(['python', 'run_regression_all_layers.py'])
    return result.returncode == 0

def run_parallel_regressions():
    """Submit parallel regression jobs"""
    print("=== Submitting Parallel Regression Jobs ===")
    
    missing_features = check_features_exist()
    if missing_features:
        print(f"Error: Feature files missing for: {missing_features}")
        print("Please run feature extraction first.")
        return False
    
    print("This will submit 14 SLURM jobs (2 models × 7 layers)")
    print("Each job will run for ~2 hours")
    print("Total estimated time: ~2 hours (parallel)")
    
    response = input("Submit parallel regression jobs? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return False
    
    result = subprocess.run(['sbatch', 'submit_fmri_predictivity_parallel.sh'])
    if result.returncode == 0:
        print("Parallel jobs submitted successfully!")
        print("Monitor progress with: squeue -u $USER")
        return True
    else:
        print("Failed to submit parallel jobs!")
        return False

def run_plotting():
    """Run plotting"""
    print("=== Running Plotting ===")
    
    # Check if regression results exist
    regression_dir = '../regression_results'
    if not os.path.exists(regression_dir):
        print("Error: Regression results directory not found")
        print("Please run regressions first.")
        return False
    
    result = subprocess.run(['python', 'plot_fmri_predictivity_all_layers.py'])
    return result.returncode == 0

def main():
    parser = argparse.ArgumentParser(description='fMRI Predictivity Analysis Workflow')
    parser.add_argument('--mode', choices=['features', 'sequential', 'parallel', 'plot', 'all'], 
                       default='interactive', help='Processing mode')
    parser.add_argument('--no-prompt', action='store_true', 
                       help='Skip confirmation prompts')
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.mode == 'interactive':
        # Interactive mode
        while True:
            print("\nSelect an option:")
            print("1. Extract features only")
            print("2. Run regressions sequentially")
            print("3. Submit parallel regression jobs")
            print("4. Run plotting")
            print("5. Run everything (features + sequential regressions + plotting)")
            print("6. Exit")
            
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                run_feature_extraction()
            elif choice == '2':
                run_sequential_regressions()
            elif choice == '3':
                run_parallel_regressions()
            elif choice == '4':
                run_plotting()
            elif choice == '5':
                print("=== Running Complete Workflow ===")
                if run_feature_extraction():
                    if run_sequential_regressions():
                        run_plotting()
            elif choice == '6':
                print("Exiting.")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
    
    else:
        # Non-interactive mode
        if args.mode == 'features':
            success = run_feature_extraction()
        elif args.mode == 'sequential':
            success = run_sequential_regressions()
        elif args.mode == 'parallel':
            success = run_parallel_regressions()
        elif args.mode == 'plot':
            success = run_plotting()
        elif args.mode == 'all':
            print("=== Running Complete Workflow ===")
            success = run_feature_extraction()
            if success:
                success = run_sequential_regressions()
                if success:
                    success = run_plotting()
        
        return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 