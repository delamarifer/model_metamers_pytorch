#!/usr/bin/env python3

import os
import subprocess
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_sound_ids(base_dir):
    """Get all sound IDs from the 3-second dataset."""
    dataset_path = os.path.join(base_dir, "3SECS_Norman-Haignere_McDermott_2018_Stimuli/NATURAL")
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Get all subdirectories (each represents a sound ID)
    sound_ids = [d for d in os.listdir(dataset_path) 
                if os.path.isdir(os.path.join(dataset_path, d))]
    return sorted(sound_ids)

def run_metamer_generation(sound_id, subclip_idx, model_type, base_dir):
    """Run metamer generation for a specific sound ID and subclip."""
    script_path = os.path.join(base_dir, "model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard/make_metamers_wsj400_behavior_only_save_metamer_layers.py")
    
    cmd = [
        "python", script_path,
        str(sound_id),
        "--duration", "3",
        "--subclip_idx", str(subclip_idx),
        "--model_type", model_type,
        "--iterations", "1000",
        "--num_rep_iter", "1",
        "--input_audio_func", "natural_sounds_norman_haignere",
        "--loss_function", "inversion_loss_layer"
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        logger.info(f"Successfully completed metamer generation for sound_id={sound_id}, subclip={subclip_idx}, model={model_type}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running metamer generation for sound_id={sound_id}, subclip={subclip_idx}, model={model_type}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Run metamer generation for all sound IDs and subclips')
    parser.add_argument('--base_dir', type=str, 
                       default="/om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch",
                       help='Base directory of the project')
    args = parser.parse_args()
    
    # Get all sound IDs
    try:
        sound_ids = get_sound_ids(args.base_dir)
        logger.info(f"Found {len(sound_ids)} sound IDs")
    except Exception as e:
        logger.error(f"Error getting sound IDs: {e}")
        return
    
    # Define subclips for 3-second duration
    subclips = [0, 1, 2]  # Corresponding to '01_04', '04_07', '07_10'
    model_types = ['standard', 'robust']
    
    # Run metamer generation for each combination
    for sound_id in sound_ids:
        for subclip_idx in subclips:
            for model_type in model_types:
                run_metamer_generation(sound_id, subclip_idx, model_type, args.base_dir)

if __name__ == "__main__":
    main() 