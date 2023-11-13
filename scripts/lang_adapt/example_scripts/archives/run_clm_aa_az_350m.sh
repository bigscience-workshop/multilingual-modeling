#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=3090-gcondo --gres=gpu:1
#SBATCH --array=1,10,100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J exp-024-run_clm_aa_az_350m

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-024/run_clm_aa_az_350m_%a.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-024/run_clm_aa_az_350m_%a.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.9.0
module load gitlfs/2.7.1
source $FP_BIGS/env_sft/bin/activate

# axis
LANG="az"
DATA_SAMPLES=$(($SLURM_ARRAY_TASK_ID * 1000))
VOCAB_SIZE=-1
BIGS_MODEL="bigscience/bloom-350m"

# adapters
EMBD_SRATEGY="original-frozen"
ADPT_STRATEGY="aa"
ADPT_CONFIG="aa"
ADPT_REDUCTION_FACTOR=16  # 16, 48, 384

tokenizer_dir="bigscience/bloom-350m"
cache_dir="/users/zyong2/data/zyong2/huggingface/"
output_dir="/users/zyong2/data/zyong2/bigscience/data/processed/024/$(basename $BIGS_MODEL)_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"
logging_dir="/users/zyong2/data/zyong2/bigscience/reports/024/$(basename $BIGS_MODEL)_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"

mkdir -p $output_dir
mkdir -p $logging_dir

python /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/lang_adapt/madx_run_clm.py \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_$LANG" \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 0.0001 \
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
    --save_steps 5000 \
    --save_strategy "steps" \
    --max_train_samples $DATA_SAMPLES \
    --max_steps 25000 \
    --logging_steps 2500 \
    --train_adapter \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --adapter_reduction_factor $ADPT_REDUCTION_FACTOR \
    --adapter_config $ADPT_CONFIG \
    --language $LANG \
    --load_best_model_at_end

# max_steps = 50000