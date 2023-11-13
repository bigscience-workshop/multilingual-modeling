#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1
# #SBATCH --constraint=v100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-024-tatoeba-pilot_baseline_labse

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-024-tatoeba/pilot_baseline_labse.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-024-tatoeba/pilot_baseline_labse.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.9.0
module load gitlfs/2.7.1
source $FP_BIGS/env_sft/bin/activate

language="bjn-en"
model_name="setu4993/LaBSE"
output_dir="/users/zyong2/data/zyong2/bigscience/data/processed/024-tatoeba/pilot_wikiann-${language}_$(basename $model_name)"
rm -rf $output_dir
mkdir -p $output_dir

python3 /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--lang_pairs $language \
--output_dir $output_dir \
--tokenizer $model_name \
--model_name $model_name \
--base_model $model_name \
--num_pairs 200 \
--seed_runs 1