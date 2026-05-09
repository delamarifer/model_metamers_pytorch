#!/usr/bin/env python3
"""
Script to generate a SLURM script for rerunning missing combinations.
This reads the missing combinations and creates an efficient array job.
"""

import os
import json
from collections import defaultdict
import argparse

def load_missing_combinations():
    """Load missing combinations from the status checker output."""
    # First run the status checker to get current missing combinations
    import subprocess
    import tempfile
    
    # Run the status checker and capture output
    result = subprocess.run(['python', 'check_combinations_status.py', '--output', 'temp_status.json'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running status checker: {}".format(result.stderr))
        return []
    
    # Load the results
    try:
        with open('temp_status.json', 'r') as f:
            data = json.load(f)
        
        # Clean up temp file
        os.remove('temp_status.json')
        
        return data['missing_combinations']
    except Exception as e:
        print("Error loading status results: {}".format(e))
        return []

def group_missing_combinations(missing_combinations):
    """
    Group missing combinations by model type and random seed for efficient array jobs.
    
    Args:
        missing_combinations: List of missing combinations
    
    Returns:
        Dictionary grouped by (model_type, random_seed)
    """
    grouped = defaultdict(lambda: {'sound_ids': set(), 'subclip_indices': set()})
    
    for combo in missing_combinations:
        key = (combo['model_type'], combo['random_seed'])
        grouped[key]['sound_ids'].add(combo['sound_id'])
        grouped[key]['subclip_indices'].add(combo['subclip_idx'])
    
    # Convert sets to sorted lists
    result = {}
    for key, data in grouped.items():
        result[key] = {
            'model_type': key[0],
            'random_seed': key[1],
            'sound_ids': sorted(list(data['sound_ids'])),
            'subclip_indices': sorted(list(data['subclip_indices']))
        }
    
    return result

def generate_slurm_script(grouped_combinations, output_file="rerun_missing_combinations.sh"):
    """
    Generate a SLURM script to rerun missing combinations.
    
    Args:
        grouped_combinations: Dictionary of grouped combinations
        output_file: Output script filename
    """
    
    script_content = """#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_missing_rerun
#SBATCH --output=output/rerun_missing_%A_%a.out
#SBATCH --error=output/rerun_missing_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-{}
#SBATCH --constraint=rocky8
#SBATCH --constraint="high-capacity&11GB"
#SBATCH --exclude=node093,node040,node094,node097,node098,node038,node037
#SBATCH --partition=normal
#SBATCH --gpu-bind=closest
#SBATCH --gpu-freq=high

# Enable error handling
set -e
set -o pipefail

# Load CUDA module if available
module load cuda70/toolkit/7.0.28

source ~/.bashrc
conda activate model_metamers_pytorch

# Get the repository root directory
REPO_ROOT="/om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch"

# Add the repository root and analysis_scripts to PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/analysis_scripts:$PYTHONPATH

# Create output directory
mkdir -p output

# Print GPU information for debugging
echo "=== GPU Information ==="
nvidia-smi
echo "======================"

# Define the combinations to run (model_type, random_seed, sound_ids, subclip_indices)
COMBINATIONS=(
""".format(len(grouped_combinations)-1)
    
    # Add each combination group
    for i, (key, data) in enumerate(grouped_combinations.items()):
        sound_ids_str = " ".join(map(str, data['sound_ids']))
        subclip_indices_str = " ".join(map(str, data['subclip_indices']))
        
        script_content += '    "{} {} {} {}"\n'.format(
            data['model_type'], 
            data['random_seed'], 
            sound_ids_str, 
            subclip_indices_str
        )
    
    script_content += """)

# Get the combination for this array task
COMBINATION_STR="${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
read -r MODEL_TYPE RANDOM_SEED SOUND_IDS_STR SUBCLIP_INDICES_STR <<< "$COMBINATION_STR"

# Convert space-separated strings to arrays
IFS=' ' read -ra SOUND_IDS <<< "$SOUND_IDS_STR"
IFS=' ' read -ra SUBCLIP_INDICES <<< "$SUBCLIP_INDICES_STR"

echo "=== Configuration ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model Type: $MODEL_TYPE"
echo "Random Seed: $RANDOM_SEED"
echo "Sound IDs: ${SOUND_IDS[*]}"
echo "Subclip Indices: ${SUBCLIP_INDICES[*]}"
echo "===================="

# Use a new unique run number for the rerun
UNIQUE_RUN_NUMBER=$((SLURM_ARRAY_JOB_ID + 2000000))  # Offset to avoid conflicts
export METAMER_RUN_NUMBER=$UNIQUE_RUN_NUMBER

echo "Using new run number: $UNIQUE_RUN_NUMBER"
echo "METAMER_RUN_NUMBER environment variable: $METAMER_RUN_NUMBER"

# Loop through sound IDs and subclip indices
for SOUND_ID in "${SOUND_IDS[@]}"; do
    for SUBCLIP_IDX in "${SUBCLIP_INDICES[@]}"; do
        echo "=== Processing sound ID $SOUND_ID, subclip $SUBCLIP_IDX ==="
        
        # Create model-specific output directory
        OUTPUT_DIR="plots/rerun_missing_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
        mkdir -p "$OUTPUT_DIR"

        echo "=== Directory Setup ==="
        echo "Output Directory: $OUTPUT_DIR"
        echo "Current Directory: $(pwd)"
        echo "======================"

        # Build the command
        CMD="python make_mms.py $SOUND_ID -I 3000 -N 8 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 1.0 --lr_decay 0.5 -R $RANDOM_SEED"

        echo "=== Running Command ==="
        echo "$CMD"
        echo "======================"

        # Run the command
        eval $CMD

        echo "=== Completed sound ID $SOUND_ID, subclip $SUBCLIP_IDX ==="
        echo ""
    done
done

echo "=== All combinations completed for array task $SLURM_ARRAY_TASK_ID ==="
"""
    
    with open(output_file, 'w') as f:
        f.write(script_content)
    
    print("Generated SLURM script: {}".format(output_file))
    print("This script will run {} array tasks".format(len(grouped_combinations)))
    
    # Print summary of what will be run
    print("\nSummary of missing combinations to be rerun:")
    total_combinations = 0
    for key, data in grouped_combinations.items():
        combinations = len(data['sound_ids']) * len(data['subclip_indices'])
        total_combinations += combinations
        print("  {} model, seed {}: {} combinations ({} sound IDs × {} subclips)".format(
            data['model_type'], data['random_seed'], combinations,
            len(data['sound_ids']), len(data['subclip_indices'])
        ))
    
    print("\nTotal combinations to rerun: {}".format(total_combinations))
    print("\nTo submit the job, run: sbatch {}".format(output_file))

def main():
    parser = argparse.ArgumentParser(description='Generate SLURM script for missing combinations')
    parser.add_argument('--output', '-o', default='rerun_missing_combinations.sh', 
                       help='Output script filename')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be run without generating script')
    
    args = parser.parse_args()
    
    print("Loading missing combinations...")
    missing_combinations = load_missing_combinations()
    
    if not missing_combinations:
        print("No missing combinations found!")
        return
    
    print("Found {} missing combinations".format(len(missing_combinations)))
    
    # Group combinations
    grouped = group_missing_combinations(missing_combinations)
    
    if args.dry_run:
        print("\nDRY RUN - Would generate script with the following groups:")
        for key, data in grouped.items():
            combinations = len(data['sound_ids']) * len(data['subclip_indices'])
            print("  {} model, seed {}: {} combinations".format(
                data['model_type'], data['random_seed'], combinations
            ))
    else:
        # Generate the SLURM script
        generate_slurm_script(grouped, args.output)

if __name__ == "__main__":
    main() 