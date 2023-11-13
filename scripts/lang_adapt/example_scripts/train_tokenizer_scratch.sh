#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-020-tokenized4clm_sampled

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-020/tokenized4clm_sampled_scratch.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-020/tokenized4clm_sampled_scratch.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
module load gitlfs/2.7.1
source $FP_BIGS/env_try_lang_adapter/bin/activate


# call by `sbatch train_tokenizer_scratch.sh my 1000 5000`
cache_dir="/users/zyong2/data/zyong2/huggingface/"
lng=$1
sample_size=$2
vocab_size=$3
MODEL="bigscience/bloom-1b3"

python /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/lang_adapt/tokenized4clm_sampled.py \
--lang $lng \
--model $MODEL \
--tokenizer_dir /users/zyong2/data/zyong2/bigscience/data/processed/020 \
--hf_cache_dir $cache_dir \
--vocab_size $vocab_size \
--sample_size $sample_size \
--use_auth_token
