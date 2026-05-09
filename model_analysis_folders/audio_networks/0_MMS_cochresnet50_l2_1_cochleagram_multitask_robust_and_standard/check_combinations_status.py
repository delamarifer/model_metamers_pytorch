#!/usr/bin/env python3
"""
Script to check which combinations are completed across all runs, regardless of run number.
This aggregates all completed combinations and shows what's missing and what's duplicated.
"""

import os
import glob
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import json

def get_all_runs(metamers_path):
    """Get all run numbers from the metamers_by_run directory."""
    run_numbers = []
    if os.path.exists(metamers_path):
        for item in os.listdir(metamers_path):
            if item.startswith("metamers_"):
                run_number = item.replace("metamers_", "")
                run_numbers.append(run_number)
    return sorted(run_numbers)

def find_completed_combinations(metamers_path, run_number):
    """
    Find all completed combinations for a specific run.
    
    Args:
        metamers_path: Path to metamers_by_run directory
        run_number: The run number to check
    
    Returns:
        List of completed combinations with run info
    """
    run_dir = os.path.join(metamers_path, "metamers_{}".format(run_number))
    
    if not os.path.exists(run_dir):
        return []
    
    completed_combinations = []
    
    # Define expected parameters
    model_types = ["robust", "standard"]
    random_seeds = [9, 400, 85]
    
    # Check each combination in this run
    for model_type in model_types:
        for random_seed in random_seeds:
            # Construct the directory name pattern
            dir_pattern = "natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS{}_I3000_N8_LR1.000_DECAY0.500_{}".format(random_seed, model_type.upper())
            model_dir = os.path.join(run_dir, dir_pattern)
            
            if not os.path.exists(model_dir):
                continue
            
            # Check each sound ID and subclip combination
            for sound_id in range(37):  # 0-36
                for subclip_idx in [0, 1, 2]:
                    # Construct the expected directory name
                    sound_dir_name = "{}_SOUND_about_{}_subclip{}".format(sound_id, model_type.upper(), subclip_idx)
                    sound_dir = os.path.join(model_dir, sound_dir_name)
                    
                    # Check if the pickle file exists
                    pickle_file = os.path.join(sound_dir, "all_metamers_pickle.pckl")
                    
                    if os.path.exists(pickle_file):
                        completed_combinations.append({
                            'sound_id': sound_id,
                            'model_type': model_type,
                            'random_seed': random_seed,
                            'subclip_idx': subclip_idx,
                            'run_number': run_number
                        })
    
    return completed_combinations

def analyze_combinations_status(metamers_path):
    """
    Analyze the status of all combinations across all runs.
    
    Args:
        metamers_path: Path to metamers_by_run directory
    
    Returns:
        Dictionary with analysis results
    """
    # Get all runs
    run_numbers = get_all_runs(metamers_path)
    print("Found {} runs: {}".format(len(run_numbers), ", ".join(run_numbers)))
    
    # Collect all completed combinations
    all_completed = []
    for run_number in run_numbers:
        completed = find_completed_combinations(metamers_path, run_number)
        all_completed.extend(completed)
        print("Run {}: {} combinations completed".format(run_number, len(completed)))
    
    # Define expected combinations
    expected_combinations = []
    sound_ids = list(range(37))  # 0-36
    model_types = ["robust", "standard"]
    random_seeds = [9, 400, 85]
    subclip_indices = [0, 1, 2]
    
    for sound_id in sound_ids:
        for model_type in model_types:
            for random_seed in random_seeds:
                for subclip_idx in subclip_indices:
                    expected_combinations.append({
                        'sound_id': sound_id,
                        'model_type': model_type,
                        'random_seed': random_seed,
                        'subclip_idx': subclip_idx
                    })
    
    # Create unique keys for comparison
    completed_keys = set()
    completed_by_key = defaultdict(list)
    
    for combo in all_completed:
        key = (combo['sound_id'], combo['model_type'], combo['random_seed'], combo['subclip_idx'])
        completed_keys.add(key)
        completed_by_key[key].append(combo['run_number'])
    
    expected_keys = set()
    for combo in expected_combinations:
        key = (combo['sound_id'], combo['model_type'], combo['random_seed'], combo['subclip_idx'])
        expected_keys.add(key)
    
    # Find missing and duplicated combinations
    missing_keys = expected_keys - completed_keys
    duplicated_keys = {key: runs for key, runs in completed_by_key.items() if len(runs) > 1}
    
    # Convert back to readable format
    missing_combinations = []
    for key in missing_keys:
        sound_id, model_type, random_seed, subclip_idx = key
        missing_combinations.append({
            'sound_id': sound_id,
            'model_type': model_type,
            'random_seed': random_seed,
            'subclip_idx': subclip_idx
        })
    
    duplicated_combinations = []
    for key, runs in duplicated_keys.items():
        sound_id, model_type, random_seed, subclip_idx = key
        duplicated_combinations.append({
            'sound_id': sound_id,
            'model_type': model_type,
            'random_seed': random_seed,
            'subclip_idx': subclip_idx,
            'runs': runs,
            'count': len(runs)
        })
    
    return {
        'total_expected': len(expected_combinations),
        'total_completed': len(completed_keys),
        'total_missing': len(missing_combinations),
        'total_duplicated': len(duplicated_combinations),
        'completion_rate': len(completed_keys) / len(expected_combinations),
        'missing_combinations': missing_combinations,
        'duplicated_combinations': duplicated_combinations,
        'completed_by_run': {run: len([c for c in all_completed if c['run_number'] == run]) for run in run_numbers}
    }

def print_summary(results):
    """Print a comprehensive summary of the analysis."""
    print("=" * 80)
    print("COMBINATIONS STATUS SUMMARY (ACROSS ALL RUNS)")
    print("=" * 80)
    
    print("\nOVERALL STATISTICS:")
    print("  Total expected combinations: {}".format(results['total_expected']))
    print("  Total completed combinations: {}".format(results['total_completed']))
    print("  Total missing combinations: {}".format(results['total_missing']))
    print("  Total duplicated combinations: {}".format(results['total_duplicated']))
    print("  Completion rate: {:.1%}".format(results['completion_rate']))
    
    print("\nCOMPLETIONS BY RUN:")
    for run, count in sorted(results['completed_by_run'].items()):
        print("  Run {}: {} combinations".format(run, count))
    
    if results['duplicated_combinations']:
        print("\nDUPLICATED COMBINATIONS (showing first 10):")
        for i, combo in enumerate(results['duplicated_combinations'][:10]):
            print("  {}. Sound {}, {}, seed {}, subclip {}: {} times (runs: {})".format(
                i+1, combo['sound_id'], combo['model_type'], combo['random_seed'], 
                combo['subclip_idx'], combo['count'], ", ".join(map(str, combo['runs']))
            ))
        if len(results['duplicated_combinations']) > 10:
            print("  ... and {} more duplicated combinations".format(len(results['duplicated_combinations']) - 10))
    
    # Analyze missing combinations by category
    if results['missing_combinations']:
        print("\nMISSING COMBINATIONS ANALYSIS:")
        
        # By model type
        model_counts = Counter(combo['model_type'] for combo in results['missing_combinations'])
        print("  Missing by model type:")
        for model_type, count in model_counts.most_common():
            print("    {}: {}".format(model_type, count))
        
        # By random seed
        seed_counts = Counter(combo['random_seed'] for combo in results['missing_combinations'])
        print("  Missing by random seed:")
        for seed, count in seed_counts.most_common():
            print("    Seed {}: {}".format(seed, count))
        
        # By sound ID
        sound_counts = Counter(combo['sound_id'] for combo in results['missing_combinations'])
        print("  Top 10 most missing sound IDs:")
        for sound_id, count in sound_counts.most_common(10):
            print("    Sound {}: {}".format(sound_id, count))
        
        # By subclip
        subclip_counts = Counter(combo['subclip_idx'] for combo in results['missing_combinations'])
        print("  Missing by subclip:")
        for subclip, count in subclip_counts.most_common():
            print("    Subclip {}: {}".format(subclip, count))

def save_results_to_json(results, output_file):
    """Save results to a JSON file for further analysis."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nDetailed results saved to: {}".format(output_file))

def main():
    parser = argparse.ArgumentParser(description='Check combinations status across all runs')
    parser.add_argument('--output', '-o', help='Save detailed results to JSON file')
    parser.add_argument('--show-missing', action='store_true', help='Show all missing combinations')
    parser.add_argument('--show-duplicates', action='store_true', help='Show all duplicated combinations')
    
    args = parser.parse_args()
    
    # Get the metamers_by_run directory
    metamers_path = "metamers_by_run"
    
    if not os.path.exists(metamers_path):
        print("Error: metamers_by_run directory not found: {}".format(metamers_path))
        return
    
    print("Analyzing combinations across all runs...")
    
    # Analyze combinations
    results = analyze_combinations_status(metamers_path)
    
    # Print summary
    print_summary(results)
    
    # Show detailed missing combinations if requested
    if args.show_missing and results['missing_combinations']:
        print("\nALL MISSING COMBINATIONS:")
        for combo in results['missing_combinations']:
            print("  Sound {}, {}, seed {}, subclip {}".format(
                combo['sound_id'], combo['model_type'], combo['random_seed'], combo['subclip_idx']
            ))
    
    # Show detailed duplicated combinations if requested
    if args.show_duplicates and results['duplicated_combinations']:
        print("\nALL DUPLICATED COMBINATIONS:")
        for combo in results['duplicated_combinations']:
            print("  Sound {}, {}, seed {}, subclip {}: {} times (runs: {})".format(
                combo['sound_id'], combo['model_type'], combo['random_seed'], 
                combo['subclip_idx'], combo['count'], ", ".join(map(str, combo['runs']))
            ))
    
    # Save to JSON if requested
    if args.output:
        save_results_to_json(results, args.output)

if __name__ == "__main__":
    main() 