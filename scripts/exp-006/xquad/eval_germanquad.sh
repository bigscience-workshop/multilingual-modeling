#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=2

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-006-eval_germanquad

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/006/eval_germanquad.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/006/eval_germanquad.err

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
    MODEL_NAME="gpt2"
    TOKENIZER_NAME="gpt2"
    OUTPUT_DIR="$FP_BIGS/data/processed/exp-006/germanquad/$lr"
    CACHE_DIR="$FP_BIGS/data/external/xquad"
    mkdir -p $OUTPUT_DIR
    
    python $FP_BIGS/scripts/exp-006/xquad/eval_qa.py \
    --output_dir $OUTPUT_DIR \
    --dataset_name "deepset/germanquad" \
    --cache_dir $CACHE_DIR \
    --num_train_epochs 50 \
    --learning_rate $lr \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name $TOKENIZER_NAME \
    --do_train \
    --do_predict
done 
