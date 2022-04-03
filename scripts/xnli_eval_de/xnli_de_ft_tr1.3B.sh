#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=1-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=2

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-013-xnli_de_ft_tr1.3B

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-013/xnli_de_ft_tr1.3B.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-013/xnli_de_ft_tr1.3B.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
source $FP_BIGS/env_lang_mod/bin/activate

# learning_rates=( 1e-5 5e-5 1e-6 5e-6 )
learning_rates=( 5e-5 )

# following https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification#fine-tuning-on-xnli
for lr in ${learning_rates[@]} ; do
    echo "LR ===== $lr"
    # MODEL_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/011/de_100000"
    MODEL_NAME="/users/zyong2/data/zyong2/huggingface/bigscience/tr5b-1B3-multilingual-alpha-checkpoints"
    # TOKENIZER_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/011/oscar-de-tokenizer"
    TOKENIZER_NAME="/users/zyong2/data/zyong2/huggingface/bigscience/tr5b-1B3-multilingual-alpha-checkpoints"
    LANG="de"
    OUTPUT_DIR="$FP_BIGS/data/processed/exp-013/xnli_${LANG}_ft_de_100000"
    CACHE_DIR="$FP_BIGS/data/external/xnli"
    mkdir -p $OUTPUT_DIR
    
    python $FP_BIGS/scripts/exp-013/xnli/xnli_de.py \
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
    --do_eval_after_train
done 
