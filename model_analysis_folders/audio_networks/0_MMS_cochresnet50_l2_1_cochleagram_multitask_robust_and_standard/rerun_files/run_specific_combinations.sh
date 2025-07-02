#!/bin/bash -l
#SBATCH --qos=normal
#SBATCH -p normal
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dlatorre@mit.edu
#SBATCH --job-name=met_specific_combinations
#SBATCH --output=output/run_specific_%A_%a.out
#SBATCH --error=output/run_specific_%A_%a.err
#SBATCH --mem=16000
#SBATCH --time=25:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --constraint=rocky8
#SBATCH --constraint="high-capacity&11GB"
#SBATCH --exclude=node093,node040,node094,node097,node098,node038,node037
#SBATCH --partition=normal
#SBATCH --gpu-bind=closest
#SBATCH --gpu-freq=high

set -e
set -o pipefail

module load cuda70/toolkit/7.0.28

source ~/.bashrc
conda activate model_metamers_pytorch

REPO_ROOT="/om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch"
export PYTHONPATH=$REPO_ROOT:$REPO_ROOT/analysis_scripts:$PYTHONPATH

mkdir -p output

echo "=== GPU Information ==="
nvidia-smi
echo "======================"

# Check if combinations file is provided
if [ -z "$COMBINATIONS_FILE" ]; then
    echo "Error: COMBINATIONS_FILE environment variable not set"
    echo "Usage: COMBINATIONS_FILE=combinations.json sbatch run_specific_combinations.sh"
    exit 1
fi

if [ ! -f "$COMBINATIONS_FILE" ]; then
    echo "Error: Combinations file $COMBINATIONS_FILE not found"
    exit 1
fi

# Get the specific combination for this array task
COMBINATION_JSON=$(python -c "
import json
import sys
with open('$COMBINATIONS_FILE', 'r') as f:
    combinations = json.load(f)
if $SLURM_ARRAY_TASK_ID < len(combinations):
    print(json.dumps(combinations[$SLURM_ARRAY_TASK_ID]))
else:
    print('{}')
")

if [ "$COMBINATION_JSON" = "{}" ]; then
    echo "No combination found for array task $SLURM_ARRAY_TASK_ID"
    exit 0
fi

# Parse the combination
MODEL_TYPE=$(echo "$COMBINATION_JSON" | python -c "import json, sys; print(json.load(sys.stdin)['model_type'])")
SOUND_ID=$(echo "$COMBINATION_JSON" | python -c "import json, sys; print(json.load(sys.stdin)['sound_id'])")
SUBCLIP_IDX=$(echo "$COMBINATION_JSON" | python -c "import json, sys; print(json.load(sys.stdin)['subclip_idx'])")
RANDOM_SEED=$(echo "$COMBINATION_JSON" | python -c "import json, sys; print(json.load(sys.stdin)['random_seed'])")

UNIQUE_RUN_NUMBER=$((SLURM_ARRAY_JOB_ID + 8000000))
export METAMER_RUN_NUMBER=$UNIQUE_RUN_NUMBER

OUTPUT_DIR="plots/run_specific_${MODEL_TYPE}_${UNIQUE_RUN_NUMBER}_seed${RANDOM_SEED}"
mkdir -p "$OUTPUT_DIR"

echo "=== Configuration ==="
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model Type: $MODEL_TYPE"
echo "Sound ID: $SOUND_ID"
echo "Subclip Index: $SUBCLIP_IDX"
echo "Random Seed: $RANDOM_SEED"
echo "Output Directory: $OUTPUT_DIR"
echo "===================="

CMD="python make_mms.py $SOUND_ID -I 3000 -N 8 -M $MODEL_TYPE -F natural_sounds_norman_haignere --duration 3 -L time_averaged_inversion_loss_layer --subclip_idx $SUBCLIP_IDX -Z 1.0 --lr_decay 0.5 -R $RANDOM_SEED"

echo "=== Running Command ==="
echo "$CMD"
echo "======================"

# Run the command
eval $CMD

echo "=== Completed sound ID $SOUND_ID, subclip $SUBCLIP_IDX, model $MODEL_TYPE, seed $RANDOM_SEED ===" 