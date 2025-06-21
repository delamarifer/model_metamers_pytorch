#!/bin/bash -l

#SBATCH --job-name=metamers_3s
#SBATCH --output=logs/metamers_3s_%A_%a.out
#SBATCH --error=logs/metamers_3s_%A_%a.err
#SBATCH --array=0-299%10  # Adjust array size based on number of combinations, run 10 at a time
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the base directory
BASE_DIR="/om2/user/dlatorre/FORKED-REPO-METAMERS/TO_COMMIT/model_metamers_pytorch"
MODEL_DIR="${BASE_DIR}/model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard"

# Get all sound IDs
SOUND_IDS=($(ls -1 "${BASE_DIR}/3SECS_Norman-Haignere_McDermott_2018_Stimuli/NATURAL/"))
NUM_SOUND_IDS=${#SOUND_IDS[@]}

# Define subclips and model types
SUBCLIPS=(0 1 2)  # Corresponding to '01_04', '04_07', '07_10'
MODEL_TYPES=("standard" "robust")

# Calculate total number of combinations
NUM_SUBCLIPS=${#SUBCLIPS[@]}
NUM_MODEL_TYPES=${#MODEL_TYPES[@]}
TOTAL_COMBINATIONS=$((NUM_SOUND_IDS * NUM_SUBCLIPS * NUM_MODEL_TYPES))

# Calculate indices for this job
SOUND_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_SUBCLIPS * NUM_MODEL_TYPES)))
SUBCLIP_IDX=$(((SLURM_ARRAY_TASK_ID / NUM_MODEL_TYPES) % NUM_SUBCLIPS))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % NUM_MODEL_TYPES))

# Get the actual values
SOUND_ID=${SOUND_IDS[$SOUND_IDX]}
SUBCLIP=${SUBCLIPS[$SUBCLIP_IDX]}
MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}

echo "Running job for:"
echo "Sound ID: $SOUND_ID"
echo "Subclip: $SUBCLIP"
echo "Model Type: $MODEL_TYPE"

# Run the metamer generation
python "${BASE_DIR}/model_analysis_folders/audio_networks/0_MMS_cochresnet50_l2_1_cochleagram_multitask_robust_and_standard/make_metamers_wsj400_behavior_only_save_metamer_layers.py" \
    "$SOUND_ID" \
    --duration 3 \
    --subclip_idx "$SUBCLIP" \
    --model_type "$MODEL_TYPE" \
    --iterations 1000 \
    --num_rep_iter 1 \
    --input_audio_func "natural_sounds_norman_haignere" \
    --loss_function "inversion_loss_layer"

# Check if the job was successful
if [ $? -eq 0 ]; then
    echo "Successfully completed metamer generation for sound_id=$SOUND_ID, subclip=$SUBCLIP, model=$MODEL_TYPE"
else
    echo "Error running metamer generation for sound_id=$SOUND_ID, subclip=$SUBCLIP, model=$MODEL_TYPE"
    exit 1
fi

# Create a log directory
mkdir -p "${BASE_DIR}/MMS_Scripts/logs"

# Function to submit a single job
submit_job() {
    local sound_id=$1
    local subclip=$2
    local model_type=$3
    
    echo "Submitting job for:"
    echo "Sound ID: $sound_id"
    echo "Subclip: $subclip"
    echo "Model Type: $model_type"
    
    # Submit the job using the existing script
    cd "$MODEL_DIR"
    sbatch --qos=normal \
           -p normal \
           --mail-type=ALL \
           --mail-user=dlatorre@mit.edu \
           --job-name=met_nhm2018_timeavg_debug \
           --output=output/robust_timeavg_debug%A_%a.out \
           --error=output/robust_timeavg_debug%A_%a.err \
           --mem=4000 \
           --time=5:00:00 \
           --gres=gpu:1 \
           --constraint=rocky8 \
           --constraint="high-capacity&11GB" \
           --exclude=node093,node040,node094,node097,node098,node038,node037 \
           --partition=normal \
           --gpu-bind=closest \
           --gpu-freq=high \
           submit_word_metamers_wsj_validation.sh "$model_type" "$subclip" > "${BASE_DIR}/MMS_Scripts/logs/job_${sound_id}_${subclip}_${model_type}.log"
    cd - > /dev/null
}

# Submit jobs for all combinations
for sound_id in "${SOUND_IDS[@]}"; do
    for subclip in "${SUBCLIPS[@]}"; do
        for model_type in "${MODEL_TYPES[@]}"; do
            submit_job "$sound_id" "$subclip" "$model_type"
            # Add a small delay to avoid overwhelming the scheduler
            sleep 1
        done
    done
done

echo "All jobs have been submitted. Check the logs directory for job IDs." 