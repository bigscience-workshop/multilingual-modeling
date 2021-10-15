#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-002-gpt2_eval_flue_paws

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/002/gpt2_eval_flue_paws.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/002/gpt2_eval_flue_paws.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $FP_BIGS/env_lang_mod/bin/activate

learning_rates=( 1e-5 5e-5 1e-6 5e-6 )
# learning_rates=( 5e-6 )
for lr in ${learning_rates[@]} ; do
    echo "LR ===== $lr"
    OUTPUT_DIR="$FP_BIGS/data/processed/exp-002/flue-paws-gpt2-base/$lr"
    MODEL_NAME="gpt2"
    mkdir -p $OUTPUT_DIR
    
    python $FP_BIGS/scripts/exp-002/eval_flue_paws.py $OUTPUT_DIR \
    --num_train_epochs 30 \
    --learning_rate $lr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --pretrained_model $MODEL_NAME \
    --tokenizer "gpt2"
done 
