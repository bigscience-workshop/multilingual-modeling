#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-013-adapters_xnli_de_tr1.3B_emb>adpt

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-013/adapters_xnli_de_tr1.3B_emb>adpt.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-013/adapters_xnli_de_tr1.3B_emb>adpt.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $FP_BIGS/env_lang_adapter/bin/activate

# learning_rates=( 1e-5 5e-5 1e-6 5e-6 )
learning_rates=( 1e-5 )
for lr in ${learning_rates[@]} ; do
    echo "LR ===== $lr"
    MODEL_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/011/de_50000"
    TOKENIZER_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/011/oscar-de-tokenizer"
    MADX_LANG_ADAPTER_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/012/madx-bigs-de-50000-50000-48/oscar_de"
    FT_STRATEGIES="task_adapters"
    LANG="de"
    OUTPUT_DIR="$FP_BIGS/data/processed/013/xnli_${LANG}_de_100K_adpt_${SLURM_ARRAY_TASK_ID}"
    CACHE_DIR="$FP_BIGS/data/external/xnli"
    mkdir -p $OUTPUT_DIR
    
    python $FP_BIGS/scripts/exp-013/xnli/adapters_xnli_de.py \
    $OUTPUT_DIR \
    --lang $LANG \
    --cache_dir $CACHE_DIR \
    --num_train_epochs 2 \
    --learning_rate $lr \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --pretrained_model $MODEL_NAME \
    --tokenizer $TOKENIZER_NAME \
    --do_train \
    --do_eval_after_train \
    --madx_lang_adapter $MADX_LANG_ADAPTER_NAME \
    --adapter_lang_name "xnli-de" \
    --finetune_strategies $FT_STRATEGIES
done 
