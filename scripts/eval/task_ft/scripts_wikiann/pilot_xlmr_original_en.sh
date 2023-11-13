#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1
# #SBATCH --constraint=v100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-024-pilot_xlmr_original_en

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-024/pilot_xlmr_original_en.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-024/pilot_xlmr_original_en.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.9.0
module load gitlfs/2.7.1
source $FP_BIGS/env_sft/bin/activate

language="en"
model_name="xlm-roberta-large"
output_dir="/users/zyong2/data/zyong2/bigscience/data/processed/024/pilot_wikiann-${language}_$(basename $model_name)"
rm -rf $output_dir
mkdir -p $output_dir

python3 /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/eval/scripts_wikiann/pilot_baseline.py \
--lang $language \
--output_dir $output_dir \
--tokenizer $model_name \
--model_name $model_name \
--base_model $model_name \
--training_config "madx" \
--seed_runs 1