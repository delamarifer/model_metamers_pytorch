import os
import sys
import subprocess
import argparse
import h5py
import time

# Settings
NUM_SPLITS = 10
RANDSEED = 3882
OVERWRITE = True
FEATURES_DIR = './features'
REGRESSION_SCRIPT = '../../../../analysis_scripts/run_regressions_all_voxels_om_natsounddata.py'

def get_layer_info(model_type):
    """Get layer information from the feature file"""
    feature_file = os.path.join(FEATURES_DIR, f'natsound_activations_{model_type}.h5')
    if not os.path.exists(feature_file):
        raise FileNotFoundError(f"Feature file not found: {feature_file}. Run extract_features_only.py first.")
    
    with h5py.File(feature_file, 'r') as h5f:
        layer_list = [layer.decode('utf-8') for layer in h5f['layer_list'][:]]
    
    return layer_list

def estimate_regression_time(layer_idx, model_type):
    """Estimate time for regression"""
    # Rough estimates based on typical processing times
    # Regression time depends on feature dimensionality and number of voxels
    feature_file = os.path.join(FEATURES_DIR, f'natsound_activations_{model_type}.h5')
    with h5py.File(feature_file, 'r') as h5f:
        layer_list = [layer.decode('utf-8') for layer in h5f['layer_list'][:]]
        if layer_idx < len(layer_list):
            layer_name = layer_list[layer_idx]
            feature_dim = h5f[layer_name].shape[1]
        else:
            return "Unknown"
    
    # Rough estimate: 1-5 minutes per layer depending on feature dimensionality
    if feature_dim < 1000:
        estimated_minutes = 1
    elif feature_dim < 5000:
        estimated_minutes = 2
    elif feature_dim < 10000:
        estimated_minutes = 3
    else:
        estimated_minutes = 5
    
    return f"{estimated_minutes}m"

def run_regression(model_type, layer_idx):
    """Run regression for a specific layer"""
    print(f"Running regression for model_type={model_type}, layer_idx={layer_idx}")
    
    # Get layer information
    layer_list = get_layer_info(model_type)
    if layer_idx >= len(layer_list):
        print(f"Error: layer_idx {layer_idx} is out of range. Available layers: 0-{len(layer_list)-1}")
        print("Available layers:")
        for i, layer in enumerate(layer_list):
            print(f"  {i}: {layer}")
        return False
    
    layer_name = layer_list[layer_idx]
    print(f"Layer name: {layer_name}")
    
    # Estimate time
    estimated_time = estimate_regression_time(layer_idx, model_type)
    print(f"Estimated regression time: {estimated_time}")
    
    # Check if feature file exists
    feature_file = os.path.join(FEATURES_DIR, f'natsound_activations_{model_type}.h5')
    if not os.path.exists(feature_file):
        print(f"Error: Feature file not found: {feature_file}")
        print("Please run extract_features_only.py first to generate features.")
        return False
    
    # Create regression output directory
    regression_dir = os.path.join('..', 'regression_results', f'natsound_activations_{model_type}')
    os.makedirs(regression_dir, exist_ok=True)
    
    # Run regression
    start_time = time.time()
    cmd = [
        'python', REGRESSION_SCRIPT,
        str(layer_idx), feature_file,
        str(NUM_SPLITS),
        str(RANDSEED),
        str(OVERWRITE)
    ]
    print('Command:', ' '.join(map(str, cmd)))
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        total_time = time.time() - start_time
        print(f"Regression completed successfully in {total_time/60:.1f} minutes")
        print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Regression failed with error code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run fMRI regression for a single layer')
    parser.add_argument('model_type', choices=['robust', 'standard'], 
                       help='Model type to analyze')
    parser.add_argument('layer_idx', type=int, 
                       help='Layer index to analyze')
    
    args = parser.parse_args()
    
    # Validate layer index
    layer_list = get_layer_info(args.model_type)
    if args.layer_idx < 0 or args.layer_idx >= len(layer_list):
        print(f"Error: layer_idx {args.layer_idx} is out of range. Available layers: 0-{len(layer_list)-1}")
        print("Available layers:")
        for i, layer in enumerate(layer_list):
            print(f"  {i}: {layer}")
        return 1
    
    # Run regression
    success = run_regression(args.model_type, args.layer_idx)
    return 0 if success else 1

if __name__ == "__main__":
    exit(main()) 