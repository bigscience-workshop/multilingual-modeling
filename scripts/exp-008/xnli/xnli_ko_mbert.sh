#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=2

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-008-xnli_ko_mbert

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/008/xnli_ko_mbert.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/008/xnli_ko_mbert.err

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
    MODEL_NAME="bert-base-multilingual-uncased"
    TOKENIZER_NAME="bert-base-multilingual-uncased"
    LANG="ko"
    OUTPUT_DIR="$FP_BIGS/data/processed/exp-008/xnli/$LANG/xnli_${LANG}_mbert_${lr}"
    CACHE_DIR="$FP_BIGS/data/external/xnli"
    mkdir -p $OUTPUT_DIR
    
    python $FP_BIGS/scripts/exp-008/xnli/xnli_ko.py \
    $OUTPUT_DIR \
    --lang $LANG \
    --cache_dir $CACHE_DIR \
    --num_train_epochs 10 \
    --learning_rate $lr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --pretrained_model $MODEL_NAME \
    --tokenizer $TOKENIZER_NAME \
    --do_train \
    --do_eval_after_train
done 
