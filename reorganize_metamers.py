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
            # Extract metadata from the directory structure
            # Example path: .../metamers/natural_sounds_norman_haignere_time_averaged_inversion_loss_layer_RS42_I3000_N8_ROBUST_20250616_022224/0_SOUND_about_ROBUST_subclip0/all_metamers_pickle.pckl
            match = re.search(r'natural_sounds_norman_haignere_(.+)_RS(\d+)_I(\d+)_N(\d+)_(\w+)_\d+_(\d+)/(\d+)_SOUND_(\w+)_(\w+)_subclip(\d+)', str(p))
            if not match:
                logging.warning("Skipping {} - directory structure not recognized".format(p))
                return

            loss_function, random_seed, iterations, num_rep_iter, model_type, timestamp, sound_id, sound_name, model_type_again, subclip_idx = match.groups()

            # Skip if this file already exists in organized directory
            organized_name = "metamer_{}_RS{}_I{}_N{}_{}_S{}_{}_subclip{}.pckl".format(
                loss_function, random_seed, iterations, num_rep_iter, model_type, sound_id, sound_name, subclip_idx)
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