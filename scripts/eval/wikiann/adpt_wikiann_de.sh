#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1
#SBATCH --array=100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-021-wikiann-adpt_wikiann_de

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-021-wikiann/adpt_wikiann_de.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-021-wikiann/adpt_wikiann_de.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
module load gitlfs/2.7.1
source $FP_BIGS/env_try_lang_adapter/bin/activate


OUTPUT_DIR="/users/zyong2/data/zyong2/bigscience/data/processed/021-wikiann/bloom-1b3-adpt-de" # where you want to save checkpoints at
LANG="de"
CACHE_DIR="/users/zyong2/data/zyong2/huggingface" # cache dir for saving/loading HF models and wikiann datasets.

LR=1e-4

BIGS_MODEL="bigscience/bloom-1b3"
ADAPTER_MODEL_DIR="/users/zyong2/data/zyong2/bigscience/data/processed/020/bloom-1b3_de_emb-and-adpt_1000samples"
TOKENIZER_NAME="bigscience/bloom-1b3"
MADX="/users/zyong2/data/zyong2/bigscience/data/processed/020/bloom-1b3_de_emb-and-adpt_1000samples/oscar_pfeiffer+inv_de"

# task-specific arguments
TASK_DATASET="wikiann"

mkdir -p $OUTPUT_DIR

python /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/eval/eval.py \
$OUTPUT_DIR \
--lang $LANG \
--cache_dir $CACHE_DIR \
--dataset $TASK_DATASET \
--num_train_epochs 2 \
--learning_rate $LR \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 1 \
--original_model $BIGS_MODEL \
--adapted_model_dir $ADAPTER_MODEL_DIR \
--madx_lang_adapter $MADX \
--tokenizer $TOKENIZER_NAME \
--do_predict \
--use_partial_data \
--use_partial_train_data 100 \
--use_partial_val_data 100 \
--use_partial_test_data 100

