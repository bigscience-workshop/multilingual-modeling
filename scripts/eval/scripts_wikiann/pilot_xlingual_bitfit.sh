#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1
# #SBATCH --constraint=v100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-025-xling-pilot_xling_bitfit

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-025-xling/pilot_xling_bitfit.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-025-xling/pilot_xling_bitfit.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
module load gitlfs/2.7.1
source $FP_BIGS/env_try_lora/bin/activate

language="de"
# ckpt="checkpoint-12500"
model_name="/users/zyong2/data/zyong2/bigscience/data/processed/024/bloom-350m_${language}_bitfit_100000samples_-1vocab_original-frozen/"
output_dir="${model_name}/pilot_wikiann-${language}"
rm -rf $output_dir
mkdir -p $output_dir

python3 /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/eval/scripts_wikiann/pilot_xlingual.py \
--lang $language \
--en_adpt_dir "/users/zyong2/data/zyong2/bigscience/data/processed/025-wikiann-en/pilot_wikiann-en_bloom-560m" \
--output_dir $output_dir \
--tokenizer "bigscience/bloom-560m" \
--model_name $model_name \
--base_model "bigscience/bloom-560m"