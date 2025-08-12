#!/bin/bash
#SBATCH --account=smearlab
#SBATCH --job-name=decode_lstm
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-10gb
#SBATCH --array=0-1
#SBATCH --mail-type=BEGIN,END,FAIL

# Print start time and allocated resources
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: $SLURM_MEM_PER_CPU"
echo "Partition: $SLURM_JOB_PARTITION"
# Optionally, if available, print additional GRES details:
if [ -n "$SLURM_GPUS" ]; then
    echo "GPUs allocated: $SLURM_GPUS"
fi


# Define mice, targets, and sessions
MICE=("6002")
BEHAVIORS=("['position_x', 'position_y', 'sns']" )
SESSIONS=("3")
WINDOW_SIZES=("0.10" "1.0")
SEQUENCE_LENGTHS=("8")




# Dimensions
NUM_MICE=${#MICE[@]}
NUM_BEHAVIORS=${#BEHAVIORS[@]}
NUM_SESSIONS=${#SESSIONS[@]}
NUM_WINDOW_SIZES=${#WINDOW_SIZES[@]}
NUM_SEQUENCE_LENGTHS=${#SEQUENCE_LENGTHS[@]}


# Total combinations per dimension
TOTAL_COMBINATIONS=$((NUM_MICE * NUM_BEHAVIORS * NUM_SESSIONS * NUM_WINDOW_SIZES * NUM_SEQUENCE_LENGTHS))

# Get indices
MOUSE_INDEX=$(( SLURM_ARRAY_TASK_ID % NUM_MICE ))
BEHAVIOR_INDEX=$(( (SLURM_ARRAY_TASK_ID / NUM_MICE) % NUM_BEHAVIORS ))
SESSION_INDEX=$(( (SLURM_ARRAY_TASK_ID / (NUM_MICE * NUM_BEHAVIORS)) % NUM_SESSIONS ))
WINDOW_INDEX=$(( (SLURM_ARRAY_TASK_ID / (NUM_MICE * NUM_BEHAVIORS * NUM_SESSIONS)) % NUM_WINDOW_SIZES ))
SEQUENCE_INDEX=$(( SLURM_ARRAY_TASK_ID / (NUM_MICE * NUM_BEHAVIORS * NUM_SESSIONS * NUM_WINDOW_SIZES) ))

# Get values
MOUSE=${MICE[$MOUSE_INDEX]}
BEHAVIOR=${BEHAVIORS[$BEHAVIOR_INDEX]}
SESSION=${SESSIONS[$SESSION_INDEX]}
WINDOW=${WINDOW_SIZES[$WINDOW_INDEX]}
SEQUENCE=${SEQUENCE_LENGTHS[$SEQUENCE_INDEX]}



# Define the data and save directories
SAVE_DIR="/projects/smearlab/shared/clickbait-mmz/figures/encoding_8_11_25"
DATA_DIR="/gpfs/ceph/ceph-smeardata/clickbait-mmz"

# Print to log
echo "Running decoding for mouse $MOUSE with behaviors $BEHAVIOR in session $SESSION with window size $WINDOW and sequence length $SEQUENCE"


# Run Python code
python LSTM_decoding.py \
    --dir "$DATA_DIR" \
    --save_dir "$SAVE_DIR" \
    --use_behaviors "$BEHAVIOR" \
    --mouse "$MOUSE" \
    --session "$SESSION" \
    --window_size "$WINDOW" \
    --step_size "$WINDOW" \
    --sigma_smooth 2.5 \
    --use_units 'good/mua' \
    --n_shifts 2 \
    --k_CV 5 \
    --n_blocks 5 \
    --plot_predictions True \
    --sequence_length "$SEQUENCE" \
    --hidden_dim 64 \
    --num_layers 2 \
    --dropout 0.1 \
    --num_epochs 1000 \
    --lr 0.001 \
    --patience 20 \
    --min_delta 0.001 \
    --factor 0.5 \
    --fs 30000 \
    --sfs 1000 \
    --target_index "-1" \
    --model_input "neural" \
    --batch_size 256
