#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=3-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-001-run_clm

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/001/run_clm.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/001/run_clm.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $FP_BIGS/env_lang_mod/bin/activate

tokenizer_dir="${FP_BIGS}/data/processed/exp-001/oscar-fr-tokenizer"
cache_dir="${FP_BIGS}/data/external/oscar_fr"
output_dir="${FP_BIGS}/data/processed/exp-001/ft-gpt2"

python $FP_BIGS/scripts/exp-001/run_clm.py \
    --model_name_or_path gpt2 \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name unshuffled_deduplicated_fr \
    --do_train \
    --do_eval \
    --output_dir $output_dir