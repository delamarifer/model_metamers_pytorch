#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import re
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

class ReorganizeMetamers:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)

    def process_pickle(self, p):
        """Process a single pickle file."""
        try:
            # Extract metadata from directory structure
            parent_dir = p.parent.parent.name
            dir_match = re.match(r'natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS(\d+)_I(\d+)_N(\d+)_(\w+)_\d+_\d+', parent_dir)
            if not dir_match:
                logging.warning("Skipping {} - directory format not recognized".format(parent_dir))
                return
                
            random_seed, iterations, num_rep_iter, model_type = dir_match.groups()
            
            # Extract sound ID from subdirectory name
            sound_dir = p.parent.name
            sound_match = re.match(r'(\d+)_SOUND_about_(\w+)_subclip(\d+)', sound_dir)
            if not sound_match:
                logging.warning("Skipping {} - sound directory format not recognized".format(sound_dir))
                return

            sound_id, sound_type, subclip_idx = sound_match.groups()
            
            # Skip if this file already exists in organized directory
            organized_name = "metamer_time_averaged_inversion_loss_layer_RS{}_I{}_N{}_{}_S{}_{}_subclip{}.pckl".format(
                random_seed, iterations, num_rep_iter, model_type, sound_id, "about", subclip_idx)
            if (self.base_dir / "metamers_organized" / organized_name).exists():
                logging.info("Skipping {} - already processed".format(p.name))
                return
                
            # Only process STANDARD model files
            if model_type != "STANDARD":
                logging.info("Skipping {} - not a STANDARD model file".format(p.name))
                return
            
            # Copy the file to the organized directory
            target_dir = self.base_dir / "metamers_organized"
            target_dir.mkdir(exist_ok=True)
            target_path = target_dir / organized_name
            
            logging.info("Copying {} to {}".format(p, target_path))
            shutil.copy2(p, target_path)
            
        except Exception as e:
            logging.error("Error processing {}: {}".format(p, str(e)))

    def process_directory(self):
        """Process all pickle files in the metamers directory."""
        metamers_dir = self.base_dir / "metamers"
        if not metamers_dir.exists():
            logging.error("Directory {} does not exist".format(metamers_dir))
            return

        # Find all pickle files in subdirectories
        pickle_files = list(metamers_dir.glob("**/all_metamers_pickle.pckl"))
        logging.info("Found {} pickle files to process".format(len(pickle_files)))

        for p in pickle_files:
            self.process_pickle(p)

def main():
    parser = argparse.ArgumentParser(description="Reorganize metamer pickle files into a flat structure")
    parser.add_argument("--base-dir", required=True, help="Base directory containing metamers folder")
    args = parser.parse_args()

    reorganizer = ReorganizeMetamers(args.base_dir)
    reorganizer.process_directory()

if __name__ == "__main__":
    main() 