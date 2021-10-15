#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-004-eval_paws_fr_swapped_embedding_ft

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/004/eval_paws_fr_swapped_embedding_ft.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/004/eval_paws_fr_swapped_embedding_ft.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $FP_BIGS/env_lang_mod/bin/activate

# learning_rates=( 1e-5 5e-5 1e-6 5e-6 )
learning_rates=( 1e-5 )
for lr in ${learning_rates[@]} ; do
    echo "LR ===== $lr"
    OUTPUT_DIR="$FP_BIGS/data/processed/exp-004/paws-fr-gpt2-rp-embedding/$lr"
    EN_MODEL_NAME="$FP_BIGS/data/processed/exp-004/paws-en-gpt2-base/1e-5/checkpoint-92610"
    FR_MODEL_NAME="$FP_BIGS/data/processed/exp-001/ft-gpt2-2/checkpoint-111500"
    TOKENIZER_NAME="$FP_BIGS/data/processed/exp-001/oscar-fr-tokenizer"
    mkdir -p $OUTPUT_DIR
    
    python $FP_BIGS/scripts/exp-004/eval_paws_fr_swapped_embedding.py $OUTPUT_DIR \
    --num_train_epochs 30 \
    --learning_rate $lr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --pretrained_model $EN_MODEL_NAME \
    --fr_gpt2_model $FR_MODEL_NAME \
    --tokenizer $TOKENIZER_NAME \
    --do_train
done 
