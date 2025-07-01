#!/usr/bin/env python3
"""
Script to check completion status across all runs in the MMS cochresnet50 folder.
This scans the metamers_by_run subdirectory to see which combinations are completed.
"""

import os
import glob
from pathlib import Path
import argparse
from collections import defaultdict, Counter
import json

def check_run_status(metamers_path, run_number):
    """
    Check the completion status for a single run.
    
    Args:
        metamers_path: Path to metamers_by_run directory
        run_number: The run number to check
    
    Returns:
        Dictionary with completion statistics for this run
    """
    run_dir = os.path.join(metamers_path, "metamers_{}".format(run_number))
    
    if not os.path.exists(run_dir):
        return {
            'run_number': run_number,
            'status': 'no_run_dir',
            'found': 0,
            'expected': 0,
            'completion_rate': 0.0,
            'missing_combinations': []
        }
    
    # Define expected parameters (based on the existing scripts)
    sound_ids = list(range(37))  # 0-36
    model_types = ["robust", "standard"]
    random_seeds = [9, 400, 85]
    subclip_indices = [0, 1, 2]
    
    total_expected = len(sound_ids) * len(model_types) * len(random_seeds) * len(subclip_indices)
    total_found = 0
    missing_combinations = []
    
    # Check each combination in this run
    for model_type in model_types:
        for random_seed in random_seeds:
            # Construct the directory name pattern
            dir_pattern = "natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS{}_I3000_N8_LR1.000_DECAY0.500_{}".format(random_seed, model_type.upper())
            model_dir = os.path.join(run_dir, dir_pattern)
            
            if not os.path.exists(model_dir):
                # Add all combinations for this model/seed as missing
                for sound_id in sound_ids:
                    for subclip_idx in subclip_indices:
                        missing_combinations.append({
                            'sound_id': sound_id,
                            'model_type': model_type,
                            'random_seed': random_seed,
                            'subclip_idx': subclip_idx
                        })
                continue
            
            # Check each sound ID and subclip combination
            for sound_id in sound_ids:
                for subclip_idx in subclip_indices:
                    # Construct the expected directory name
                    sound_dir_name = "{}_SOUND_about_{}_subclip{}".format(sound_id, model_type.upper(), subclip_idx)
                    sound_dir = os.path.join(model_dir, sound_dir_name)
                    
                    # Check if the pickle file exists
                    pickle_file = os.path.join(sound_dir, "all_metamers_pickle.pckl")
                    
                    if os.path.exists(pickle_file):
                        total_found += 1
                    else:
                        missing_combinations.append({
                            'sound_id': sound_id,
                            'model_type': model_type,
                            'random_seed': random_seed,
                            'subclip_idx': subclip_idx
                        })
    
    completion_rate = total_found / total_expected if total_expected > 0 else 0.0
    
    return {
        'run_number': run_number,
        'status': 'has_data',
        'found': total_found,
        'expected': total_expected,
        'completion_rate': completion_rate,
        'missing_combinations': missing_combinations
    }

def get_all_runs(metamers_path):
    """Get all run numbers from the metamers_by_run directory."""
    run_numbers = []
    if os.path.exists(metamers_path):
        for item in os.listdir(metamers_path):
            if item.startswith("metamers_"):
                run_number = item.replace("metamers_", "")
                run_numbers.append(run_number)
    return sorted(run_numbers)

def print_summary(all_results, verbose=False):
    """Print a summary of all run results."""
    print("=" * 80)
    print("COMPLETION STATUS SUMMARY FOR MMS COCHRESNET50 FOLDER")
    print("=" * 80)
    
    # Overall statistics
    total_runs = len(all_results)
    runs_with_data = sum(1 for r in all_results if r['status'] == 'has_data')
    runs_with_no_data = sum(1 for r in all_results if r['status'] == 'no_run_dir')
    
    print(f"\nOVERALL STATISTICS:")
    print(f"  Total runs found: {total_runs}")
    print(f"  Runs with data: {runs_with_data}")
    print(f"  Runs with no data: {runs_with_no_data}")
    
    if runs_with_data > 0:
        completion_rates = [r['completion_rate'] for r in all_results if r['status'] == 'has_data']
        avg_completion = sum(completion_rates) / len(completion_rates)
        total_expected = sum(r['expected'] for r in all_results if r['status'] == 'has_data')
        total_found = sum(r['found'] for r in all_results if r['status'] == 'has_data')
        
        print("  Average completion rate: {:.1%}".format(avg_completion))
        print("  Total combinations found: {}".format(total_found))
        print("  Total combinations expected: {}".format(total_expected))
        print("  Overall completion rate: {:.1%}".format(total_found/total_expected))
        
        # Find best and worst performing runs
        completed_runs = [r for r in all_results if r['status'] == 'has_data']
        completed_runs.sort(key=lambda x: x['completion_rate'], reverse=True)
        
        print(f"\nTOP 5 MOST COMPLETE RUNS:")
        for i, result in enumerate(completed_runs[:5]):
            print(f"  {i+1}. Run {result['run_number']}: {result['completion_rate']:.1%} ({result['found']}/{result['expected']})")
        
        print(f"\nBOTTOM 5 LEAST COMPLETE RUNS:")
        for i, result in enumerate(completed_runs[-5:]):
            print(f"  {i+1}. Run {result['run_number']}: {result['completion_rate']:.1%} ({result['found']}/{result['expected']})")
    
    # Detailed results if verbose
    if verbose:
        print(f"\nDETAILED RESULTS:")
        for result in all_results:
            print(f"\nRun {result['run_number']}:")
            if result['status'] == 'no_run_dir':
                print(f"  Status: No run directory found")
            else:
                print(f"  Status: {result['found']}/{result['expected']} combinations completed ({result['completion_rate']:.1%})")
                if result['missing_combinations']:
                    print(f"  Missing combinations: {len(result['missing_combinations'])}")
                    # Show first few missing combinations
                    for i, combo in enumerate(result['missing_combinations'][:5]):
                        print(f"    - Sound {combo['sound_id']}, {combo['model_type']}, seed {combo['random_seed']}, subclip {combo['subclip_idx']}")
                    if len(result['missing_combinations']) > 5:
                        print(f"    ... and {len(result['missing_combinations']) - 5} more")

def analyze_missing_combinations(all_results):
    """Analyze missing combinations across all runs to find patterns."""
    print(f"\nMISSING COMBINATIONS ANALYSIS:")
    print("=" * 50)
    
    # Collect all missing combinations
    all_missing = []
    for result in all_results:
        if result['status'] == 'has_data':
            all_missing.extend(result['missing_combinations'])
    
    if not all_missing:
        print("No missing combinations found!")
        return
    
    # Count by model type
    model_counts = Counter(combo['model_type'] for combo in all_missing)
    print(f"\nMissing by model type:")
    for model_type, count in model_counts.most_common():
        print(f"  {model_type}: {count}")
    
    # Count by random seed
    seed_counts = Counter(combo['random_seed'] for combo in all_missing)
    print(f"\nMissing by random seed:")
    for seed, count in seed_counts.most_common():
        print(f"  Seed {seed}: {count}")
    
    # Count by sound ID
    sound_counts = Counter(combo['sound_id'] for combo in all_missing)
    print(f"\nTop 10 most missing sound IDs:")
    for sound_id, count in sound_counts.most_common(10):
        print(f"  Sound {sound_id}: {count}")
    
    # Count by subclip
    subclip_counts = Counter(combo['subclip_idx'] for combo in all_missing)
    print(f"\nMissing by subclip:")
    for subclip, count in subclip_counts.most_common():
        print(f"  Subclip {subclip}: {count}")

def save_results_to_json(all_results, output_file):
    """Save results to a JSON file for further analysis."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Check completion status across all runs in MMS cochresnet50 folder')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print detailed results')
    parser.add_argument('--output', '-o', help='Save detailed results to JSON file')
    parser.add_argument('--analyze-missing', action='store_true', help='Analyze missing combinations patterns')
    
    args = parser.parse_args()
    
    # Get the metamers_by_run directory
    metamers_path = "metamers_by_run"
    
    if not os.path.exists(metamers_path):
        print(f"Error: metamers_by_run directory not found: {metamers_path}")
        return
    
    print(f"Scanning metamers_by_run directory: {metamers_path}")
    
    # Get all runs
    run_numbers = get_all_runs(metamers_path)
    print(f"Found {len(run_numbers)} runs:")
    for run_num in run_numbers:
        print(f"  - {run_num}")
    
    # Check each run
    all_results = []
    for run_number in run_numbers:
        result = check_run_status(metamers_path, run_number)
        all_results.append(result)
    
    # Print summary
    print_summary(all_results, verbose=args.verbose)
    
    # Analyze missing combinations if requested
    if args.analyze_missing:
        analyze_missing_combinations(all_results)
    
    # Save to JSON if requested
    if args.output:
        save_results_to_json(all_results, args.output)

if __name__ == "__main__":
    main() 