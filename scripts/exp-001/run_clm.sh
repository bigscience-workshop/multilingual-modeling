#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=5-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:8

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=16

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
output_dir="${FP_BIGS}/data/processed/exp-001/ft-gpt2-2"
logging_dir="${FP_BIGS}/reports/exp-001/ft-gpt2-2"
ckpt_dir="${FP_BIGS}/data/processed/exp-001/ft-gpt2-2/checkpoint-25000"

python $FP_BIGS/scripts/exp-001/run_clm.py \
    --model_name_or_path gpt2 \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name unshuffled_deduplicated_fr \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 0.001 \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 4 \
    --eval_steps 1000 \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --resume_from_checkpoint $ckpt_dir