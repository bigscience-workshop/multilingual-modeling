#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=3-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-031-download_oscar_ko

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/001/download_oscar_ko.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/001/download_oscar_ko.err

# Set up the environment by loading modules
# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $FP_BIGS/env_lang_mod/bin/activate

python3 $FP_BIGS/scripts/exp-001/download_oscar_ko.py