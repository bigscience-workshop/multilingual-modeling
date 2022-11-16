#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1
#SBATCH --array=100

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=4

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-030-run_clm_sft_ru_560m

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-030/run_clm_sft_ru_560m_%a.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-030/run_clm_sft_ru_560m_%a.err

# Set up the environment by loading modules
set -a # automatically export all variables
source ~/.env
set +a

module load python/3.9.0
module load gitlfs/2.7.1
# module load cuda/11.1.1
# module load gcc/10.2
source $FP_BIGS/env_sft/bin/activate

# axis
LANG="ru"
DATA_SAMPLES=$(($SLURM_ARRAY_TASK_ID * 1000))
VOCAB_SIZE=-1
BIGS_MODEL="bigscience/bloom-560m"

# adapters
EMBD_SRATEGY="original"
ADPT_STRATEGY="sft"
ft_params_num=3171840

tokenizer_dir="bigscience/bloom-560m"
cache_dir="/users/zyong2/data/zyong2/huggingface/"
output_dir="/users/zyong2/data/zyong2/bigscience/data/processed/030/$(basename $BIGS_MODEL)_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"
logging_dir="/users/zyong2/data/zyong2/bigscience/reports/030/$(basename $BIGS_MODEL)_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"

mkdir -p $output_dir
mkdir -p $logging_dir

# deepspeed --num_gpus=1 --master_port 60000 \
python3 /users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/lang_adapt/madx_run_clm.py \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_$LANG" \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 0.00005 \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 4 \
    --eval_steps 2500 \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --save_steps 2500 \
    --save_strategy "steps" \
    --max_train_samples $DATA_SAMPLES \
    --logging_steps 2500 \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --language $LANG \
    --load_best_model_at_end \
    --train_sft \
    --ft_params_num $ft_params_num \
    --full_ft_max_steps_per_iteration 12500 \
    --sparse_ft_max_steps_per_iteration 12500 \
    --fp16 \
    --gradient_checkpointing \
    --tie_word_embeddings 0 
    # --deepspeed "/users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/scripts/lang_adapt/ds_config_zero2.json"


# max_steps = 50000