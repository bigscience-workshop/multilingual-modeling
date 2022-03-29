#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1
#SBATCH --array=100,200,500

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-020-run_clm_emb

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-020/run_clm_emb_%a.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-020/run_clm_emb_%a.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.7.4
module load gitlfs/2.7.1
source $FP_BIGS/env_lang_adapter/bin/activate


# axis
LANG="th"
MAX_TRAIN_SAMPLES=$(($SLURM_ARRAY_TASK_ID * 1000))
BIGS_MODEL="/users/zyong2/data/zyong2/huggingface/bigscience/tr5b-1B3-multilingual-alpha-checkpoints"


tokenizer_dir="/users/zyong2/data/zyong2/bigscience/data/processed/020/th_oscar_tokenizer_full"
cache_dir="/users/zyong2/data/zyong2/huggingface/"
output_dir="/users/zyong2/data/zyong2/bigscience/data/processed/020/${LANG}_emb_${MAX_TRAIN_SAMPLES}samples"
logging_dir="/users/zyong2/data/zyong2/bigscience/data/reports/020/${LANG}_emb_${MAX_TRAIN_SAMPLES}samples"
mkdir -p $output_dir
mkdir -p $logging_dir

python $FP_BIGS/scripts/exp-020/madx_run_clm.py \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_$LANG" \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 0.001 \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 4 \
    --eval_steps 5000 \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --save_steps 25000 \
    --save_strategy "steps" \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_steps 50000 \
    --lang_adapt_strategies "emb" \
    --embedding_strategies "replace"