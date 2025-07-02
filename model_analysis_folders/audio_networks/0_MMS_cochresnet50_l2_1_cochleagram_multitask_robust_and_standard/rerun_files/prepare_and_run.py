#!/usr/bin/env python3
"""
General script to prepare and run specific combinations.
Can handle any random seed, model type, or specific combinations.
"""

import os
import json
import subprocess
import argparse
from pathlib import Path

def load_missing_combinations():
    """Load missing combinations from the status checker."""
    result = subprocess.run(['python', 'check_combinations_status.py', '--output', 'temp_status.json'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error running status checker: {}".format(result.stderr))
        return []
    
    try:
        with open('temp_status.json', 'r') as f:
            data = json.load(f)
        os.remove('temp_status.json')
        return data['missing_combinations']
    except Exception as e:
        print("Error loading status results: {}".format(e))
        return []

def filter_combinations(combinations, random_seed=None, model_type=None, sound_ids=None, subclip_indices=None):
    """Filter combinations based on criteria."""
    filtered = combinations
    
    if random_seed is not None:
        filtered = [c for c in filtered if c['random_seed'] == random_seed]
    
    if model_type is not None:
        filtered = [c for c in filtered if c['model_type'] == model_type]
    
    if sound_ids is not None:
        filtered = [c for c in filtered if c['sound_id'] in sound_ids]
    
    if subclip_indices is not None:
        filtered = [c for c in filtered if c['subclip_idx'] in subclip_indices]
    
    return filtered

def create_combinations_file(combinations, output_file):
    """Save combinations to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(combinations, f, indent=2)
    print(f"Saved {len(combinations)} combinations to {output_file}")

def update_slurm_array_size(array_size):
    """Update the SLURM script array size."""
    script_path = "run_specific_combinations.sh"
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Update the array size
    import re
    content = re.sub(r'#SBATCH --array=0-\d+', f'#SBATCH --array=0-{array_size-1}', content)
    
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Updated SLURM script array size to 0-{array_size-1}")

def run_slurm_job(combinations_file):
    """Submit the SLURM job."""
    cmd = f"COMBINATIONS_FILE={combinations_file} sbatch run_specific_combinations.sh"
    print(f"Running: {cmd}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Job submitted successfully!")
        print(result.stdout)
    else:
        print("Error submitting job:")
        print(result.stderr)

def main():
    parser = argparse.ArgumentParser(description='Prepare and run specific combinations')
    parser.add_argument('--random-seed', type=int, help='Filter by random seed')
    parser.add_argument('--model-type', choices=['standard', 'robust'], help='Filter by model type')
    parser.add_argument('--sound-ids', type=int, nargs='+', help='Filter by specific sound IDs')
    parser.add_argument('--subclip-indices', type=int, nargs='+', help='Filter by specific subclip indices')
    parser.add_argument('--combinations-file', help='Use existing combinations file')
    parser.add_argument('--output-file', default='combinations.json', help='Output combinations file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without submitting')
    parser.add_argument('--submit', action='store_true', help='Submit the SLURM job after preparation')
    
    args = parser.parse_args()
    
    if args.combinations_file:
        # Use existing combinations file
        with open(args.combinations_file, 'r') as f:
            combinations = json.load(f)
        print(f"Loaded {len(combinations)} combinations from {args.combinations_file}")
    else:
        # Load missing combinations and filter
        print("Loading missing combinations...")
        combinations = load_missing_combinations()
        
        if not combinations:
            print("No missing combinations found!")
            return
        
        print(f"Found {len(combinations)} total missing combinations")
        
        # Apply filters
        combinations = filter_combinations(
            combinations, 
            random_seed=args.random_seed,
            model_type=args.model_type,
            sound_ids=args.sound_ids,
            subclip_indices=args.subclip_indices
        )
    
    if not combinations:
        print("No combinations match the specified criteria!")
        return
    
    print(f"Filtered to {len(combinations)} combinations")
    
    # Show summary
    print("\nCombinations summary:")
    by_seed = {}
    by_model = {}
    for combo in combinations:
        seed = combo['random_seed']
        model = combo['model_type']
        by_seed[seed] = by_seed.get(seed, 0) + 1
        by_model[model] = by_model.get(model, 0) + 1
    
    print("By random seed:")
    for seed, count in sorted(by_seed.items()):
        print(f"  Seed {seed}: {count}")
    
    print("By model type:")
    for model, count in sorted(by_model.items()):
        print(f"  {model}: {count}")
    
    # Create combinations file
    create_combinations_file(combinations, args.output_file)
    
    # Update SLURM script array size
    update_slurm_array_size(len(combinations))
    
    if args.dry_run:
        print(f"\nDRY RUN - Would submit job with {len(combinations)} array tasks")
        print(f"Combinations file: {args.output_file}")
    elif args.submit:
        run_slurm_job(args.output_file)
    else:
        print(f"\nReady to submit job with {len(combinations)} array tasks")
        print(f"Combinations file: {args.output_file}")
        print("Run with: --submit to submit the job")

if __name__ == "__main__":
    main() 