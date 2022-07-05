#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=cpu 

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g



lng=$1
model=$2
tokenizer_dir=$3
vocab_size=$4
sample_size=$5
python tokenized4clm_sampled.py --lang $lng --model $model --tokenizer_dir $tokenizer_dir --vocab_size $vocab_size --sample_size $sample_size --extend_vocab

