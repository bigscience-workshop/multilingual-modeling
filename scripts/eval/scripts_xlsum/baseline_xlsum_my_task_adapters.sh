#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-021-xlsum-baseline_xlsum_my_task_adapters

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-021-xlsum/baseline_xlsum_my_task_adapters.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-021-xlsum/baseline_xlsum_my_task_adapters.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
module load gitlfs/2.7.1
source $FP_BIGS/env_try_lang_adapter/bin/activate


LR=1e-5

BIGS_MODEL="bigscience/bloom-1b3"
MODEL_NAME="bigscience/bloom-1b3"
TOKENIZER_NAME="bigscience/bloom-1b3"

# task-specific arguments
TASK_DATASET="csebuetnlp/xlsum"
TASK_LAYER="task-adapters"
LANG="burmese"
OUTPUT_DIR="/users/zyong2/data/zyong2/bigscience/data/processed/021-xlsum/$(basename $BIGS_MODEL)-baseline-${LANG}-FT-${TASK_LAYER}" # where you want to save checkpoints at
CACHE_DIR="/users/zyong2/data/zyong2/huggingface" # cache dir for saving/loading HF models and XNLI datasets.


mkdir -p $OUTPUT_DIR

python /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/eval/eval.py \
$OUTPUT_DIR \
--lang $LANG \
--cache_dir $CACHE_DIR \
--dataset $TASK_DATASET \
--num_train_epochs 10 \
--learning_rate $LR \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--original_model $BIGS_MODEL \
--adapted_model_dir $MODEL_NAME \
--tokenizer $TOKENIZER_NAME \
--do_train \
--do_predict \
--task_layers $TASK_LAYER \
--baseline
