#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_nhm2018
#SBATCH --output=output/robust%A_%a.out
#SBATCH --error=output/robust%A_%a.err
#SBATCH --mem=4000
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0  # Adjust this based on number of files you want to process
#SBATCH --constraint=rocky8
#SBATCH --constraint="high-capacity&11GB"
#SBATCH --exclude=node093,node040,node094,node097,node098,node038,node037
#SBATCH --partition=normal
#SBATCH --gpu-bind=closest
#SBATCH --gpu-freq=high

# Load CUDA module if available
module load cuda70/toolkit/7.0.28

source ~/.bashrc
conda activate model_metamers_pytorch

# Get the repository root directory
REPO_ROOT="/om2/user/dlatorre/FORKED-REPO-METAMERS/model_metamers_pytorch"

# Add the repository root to PYTHONPATH
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

# Print GPU information for debugging
nvidia-smi

# Default to robust model if not specified
MODEL_TYPE=${1:-robust}

# Validate model type
if [[ "$MODEL_TYPE" != "robust" && "$MODEL_TYPE" != "standard" ]]; then
    echo "Error: Model type must be either 'robust' or 'standard'"
    exit 1
fi

# Run the metamer generation script with the specified model type
python make_metamers_wsj400_behavior_only_save_metamer_layers.py $SLURM_ARRAY_TASK_ID -I 1 -N 1 -M $MODEL_TYPE -F norman_haignere_mcdermott_2018

# First evaluate the model's performance
# echo "Evaluating model performance..."
# python evaluate_word_classification.py > output/evaluation_results.txt

# # Check if accuracy is above 80%
# if grep -q "Model performance is above 80%" output/evaluation_results.txt; then
#     echo "Model performance is good. Proceeding with metamer generation..."
    # Run the metamer generation script with the Norman-Haignere McDermott 2018 dataset
# python make_metamers_wsj400_behavior_only_save_metamer_layers.py $SLURM_ARRAY_TASK_ID  -I 1 -N 1
# else
#     echo "Model performance is below 80%. Stopping execution."
#     exit 1
# fi 