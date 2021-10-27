#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=5-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:8

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-001-run_clm_no_tok

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/001/run_clm_no_tok.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/001/run_clm_no_tok.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $FP_BIGS/env_lang_mod/bin/activate

tokenizer_dir="${FP_BIGS}/data/processed/exp-001/oscar-fr-tokenizer"
cache_dir="${FP_BIGS}/data/external/oscar_fr"
output_dir="${FP_BIGS}/data/processed/exp-001/ft-gpt2-no-tok"

python $FP_BIGS/scripts/exp-001/run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name unshuffled_deduplicated_fr \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 4 \
    --eval_steps 500 \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000