#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=cpu 

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g



bs_dir=/tmp-network/user/vnikouli/Projects/bigscience
lng=$1
sample_size=$2
vocab_size=$3
source $bs_dir/multilingual-modeling/scripts/env/bin/activate
python tokenized4clm_sampled.py --lang $lng --tokenizer_dir $bs_dir/tokenizers --hf_cache_dir $bs_dir/data --vocab_size $vocab_size --sample_size $sample_size --replace_with_overlap

