#!/usr/bin/env python3
"""
Script to generate a SLURM script for rerunning missing combinations with higher parallelism.
This creates more array tasks to better utilize the 16 parallel allocation.
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

def group_missing_combinations_parallel(missing_combinations, target_tasks=16):
    """
    Group missing combinations into smaller chunks for higher parallelism.
    
    Args:
        missing_combinations: List of missing combinations
        target_tasks: Target number of array tasks (default 16)
    
    Returns:
        List of grouped combinations for array tasks
    """
    # First group by model type and random seed
    grouped_by_model_seed = defaultdict(lambda: {'sound_ids': set(), 'subclip_indices': set()})
    
    for combo in missing_combinations:
        key = (combo['model_type'], combo['random_seed'])
        grouped_by_model_seed[key]['sound_ids'].add(combo['sound_id'])
        grouped_by_model_seed[key]['subclip_indices'].add(combo['subclip_idx'])
    
    # Convert to list format
    model_seed_groups = []
    for key, data in grouped_by_model_seed.items():
        model_seed_groups.append({
            'model_type': key[0],
            'random_seed': key[1],
            'sound_ids': sorted(list(data['sound_ids'])),
            'subclip_indices': sorted(list(data['subclip_indices']))
        })
    
    # Now split into smaller chunks
    parallel_groups = []
    
    for group in model_seed_groups:
        sound_ids = group['sound_ids']
        subclip_indices = group['subclip_indices']
        
        # Calculate how many combinations this group has
        total_combinations = len(sound_ids) * len(subclip_indices)
        
        # If this group is small enough, keep it as one task
        if total_combinations <= 8:  # Small groups stay as one task
            parallel_groups.append(group)
        else:
            # Split larger groups by sound IDs
            # Try to create chunks of roughly equal size
            chunk_size = max(1, len(sound_ids) // 2)  # Split into at least 2 chunks
            
            for i in range(0, len(sound_ids), chunk_size):
                chunk_sound_ids = sound_ids[i:i + chunk_size]
                parallel_groups.append({
                    'model_type': group['model_type'],
                    'random_seed': group['random_seed'],
                    'sound_ids': chunk_sound_ids,
                    'subclip_indices': subclip_indices
                })
    
    # If we still have too few tasks, split further
    if len(parallel_groups) < target_tasks:
        # Split the largest groups further
        parallel_groups.sort(key=lambda x: len(x['sound_ids']) * len(x['subclip_indices']), reverse=True)
        
        final_groups = []
        for group in parallel_groups:
            sound_ids = group['sound_ids']
            subclip_indices = group['subclip_indices']
            total_combinations = len(sound_ids) * len(subclip_indices)
            
            if total_combinations > 6 and len(final_groups) < target_tasks - 1:
                # Split this group further
                mid_point = len(sound_ids) // 2
                final_groups.append({
                    'model_type': group['model_type'],
                    'random_seed': group['random_seed'],
                    'sound_ids': sound_ids[:mid_point],
                    'subclip_indices': subclip_indices
                })
                final_groups.append({
                    'model_type': group['model_type'],
                    'random_seed': group['random_seed'],
                    'sound_ids': sound_ids[mid_point:],
                    'subclip_indices': subclip_indices
                })
            else:
                final_groups.append(group)
        
        parallel_groups = final_groups
    
    return parallel_groups

def generate_slurm_script_parallel(grouped_combinations, output_file="rerun_missing_combinations_parallel.sh"):
    """
    Generate a SLURM script to rerun missing combinations with higher parallelism.
    
    Args:
        grouped_combinations: List of grouped combinations
        output_file: Output script filename
    """
    
    script_content = """#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_missing_parallel
#SBATCH --output=output/rerun_missing_parallel_%A_%a.out
#SBATCH --error=output/rerun_missing_parallel_%A_%a.err
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
    for data in grouped_combinations:
        sound_ids_str = " ".join(map(str, data['sound_ids']))
        subclip_indices_str = " ".join(map(str, data['subclip_indices']))
        
        script_content += '    "{}|{}|{}|{}"\n'.format(
            data['model_type'], 
            data['random_seed'], 
            sound_ids_str, 
            subclip_indices_str
        )
    
    script_content += """)

# Get the combination for this array task
COMBINATION_STR="${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"

# Split by pipe character to avoid issues with spaces in sound_ids
IFS='|' read -r MODEL_TYPE RANDOM_SEED SOUND_IDS_STR SUBCLIP_INDICES_STR <<< "$COMBINATION_STR"

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
UNIQUE_RUN_NUMBER=$((SLURM_ARRAY_JOB_ID + 3000000))  # Different offset for parallel version
export METAMER_RUN_NUMBER=$UNIQUE_RUN_NUMBER

echo "Using new run number: $UNIQUE_RUN_NUMBER"
echo "METAMER_RUN_NUMBER environment variable: $METAMER_RUN_NUMBER"

# Loop through sound IDs and subclip indices
for SOUND_ID in "${SOUND_IDS[@]}"; do
    for SUBCLIP_IDX in "${SUBCLIP_INDICES[@]}"; do
        echo "=== Processing sound ID $SOUND_ID, subclip $SUBCLIP_IDX ==="
        
        # Create model-specific output directory
        OUTPUT_DIR="plots/rerun_missing_parallel_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
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
    
    print("Generated parallel SLURM script: {}".format(output_file))
    print("This script will run {} array tasks".format(len(grouped_combinations)))
    
    # Print summary of what will be run
    print("\nSummary of parallel tasks:")
    total_combinations = 0
    for i, data in enumerate(grouped_combinations):
        combinations = len(data['sound_ids']) * len(data['subclip_indices'])
        total_combinations += combinations
        print("  Task {}: {} model, seed {}: {} combinations ({} sound IDs × {} subclips)".format(
            i, data['model_type'], data['random_seed'], combinations,
            len(data['sound_ids']), len(data['subclip_indices'])
        ))
    
    print("\nTotal combinations to rerun: {}".format(total_combinations))
    print("Average combinations per task: {:.1f}".format(total_combinations / len(grouped_combinations)))
    print("\nTo submit the job, run: sbatch {}".format(output_file))

def main():
    parser = argparse.ArgumentParser(description='Generate parallel SLURM script for missing combinations')
    parser.add_argument('--output', '-o', default='rerun_missing_combinations_parallel.sh', 
                       help='Output script filename')
    parser.add_argument('--target-tasks', '-t', type=int, default=16,
                       help='Target number of array tasks (default: 16)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be run without generating script')
    
    args = parser.parse_args()
    
    print("Loading missing combinations...")
    missing_combinations = load_missing_combinations()
    
    if not missing_combinations:
        print("No missing combinations found!")
        return
    
    print("Found {} missing combinations".format(len(missing_combinations)))
    
    # Group combinations for higher parallelism
    grouped = group_missing_combinations_parallel(missing_combinations, args.target_tasks)
    
    if args.dry_run:
        print("\nDRY RUN - Would generate parallel script with {} tasks:".format(len(grouped)))
        total_combinations = 0
        for i, data in enumerate(grouped):
            combinations = len(data['sound_ids']) * len(data['subclip_indices'])
            total_combinations += combinations
            print("  Task {}: {} model, seed {}: {} combinations".format(
                i, data['model_type'], data['random_seed'], combinations
            ))
        print("Total combinations: {}".format(total_combinations))
        print("Average per task: {:.1f}".format(total_combinations / len(grouped)))
    else:
        # Generate the SLURM script
        generate_slurm_script_parallel(grouped, args.output)

if __name__ == "__main__":
    main() 