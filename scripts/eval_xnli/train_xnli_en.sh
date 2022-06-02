#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu
#SBATCH --gres="gpu:1"

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J run_clm_madx

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vassilina.nikoulina@naverlabs.com
#SBATCH --constraint="gpu_v100&gpu_32g"

# XNLI (Cross-Lingual and Supervised Setting)

FP_BIGS=/tmp-network/user/vnikouli/Projects/bigscience
# Set up the environment by loading modules
source $FP_BIGS/multilingual-modeling/scripts/env/bin/activate
ch=118500
model_name="tr5b-1B3-multilingual-alpha-checkpoints/ch${ch}"
ORIGINAL_MODEL=${FP_BIGS}/multilingual-modeling/scripts/exp-009/$model_name 
CACHE_DIR="${FP_BIGS}/data/"
# we finetune task adapters for XNLI
FT_STRATEGIES="task_adapters"


LR=1e-5

OUTPUT_DIR=$ORIGINAL_MODEL/xnli_task_adapter_full_${FT_STRATEGIES}_fp16
mkdir -p $OUTPUT_DIR
python adapters_xnli_de_vn.py \
$OUTPUT_DIR \
--lang en \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $LR \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--original_model $ORIGINAL_MODEL \
--do_train \
--do_eval_after_train \
--finetune_strategies ${FT_STRATEGIES} \
--cross_lingual &> $OUTPUT_DIR/train.log



