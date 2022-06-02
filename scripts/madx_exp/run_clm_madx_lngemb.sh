#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu
#SBATCH --gres="gpu:1"

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=16

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J exp-009-run_clm_de_madx

# Specify an output file
#SBATCH -o /tmp-network/user/vnikouli/Projects/bigscience/logs/run_clm_de_madx-%j.out
#SBATCH -e /tmp-network/user/vnikouli/Projects/bigscience/logs/run_clm_de_madx-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vassilina.nikoulina@naverlabs.com

# Set up the environment by loading modules
source /tmp-network/user/vnikouli/Projects/bigscience/multilingual-modeling/scripts/env/bin/activate
FP_BIGS=/tmp-network/user/vnikouli/Projects/bigscience/

data_sample=100000
ch=$1
lng=$2
adapter_reduction_factor=$3
dataset=oscar
adapter_config="pfeiffer+inv"

model_name="tr5b-1B3-multilingual-alpha-checkpoints/ch${ch}"
tokenizer_dir="${FP_BIGS}/tokenizers/bigscience-1.3B-${lng}-tokenizer"
cache_dir="${FP_BIGS}/data/${dataset}_${lng}"
data_dir="${FP_BIGS}/exp-009/madx-bs1b3-multi-ch${ch}-${lng}-sample${data_sample}"
data_tok_dir="${FP_BIGS}/exp-009/madx-bs1b3-multi-ch${ch}-${lng}-sample${data_sample}/lng_tok"
output_dir="${data_dir}/withlngembft-lmhead-${adapter_config}-${adapter_reduction_factor}"
logging_dir="${FP_BIGS}/logs/exp-009/madx-bs1b3-multi-ch${ch}-${dataset}-${lng}-sample${data_sample}-withlngembft-lmhead-${adapter_config}-${adapter_reduction_factor}"


python $FP_BIGS/multilingual-modeling/scripts/madx_exp/madx_lngembft_clm.py \
    --fp16 \
    --model_name_or_path ${FP_BIGS}/multilingual-modeling/scripts/exp-009/$model_name \
    --tokenizer_name ${tokenizer_dir} \
    --dataset_name ${dataset} \
    --cache_dir $cache_dir \
    --dataset_config_name unshuffled_deduplicated_${lng} \
    --logging_dir ${logging_dir} \
    --report_to "tensorboard" \
    --learning_rate 0.001 \
    --do_train \
    --do_eval \
    --output_dir ${output_dir} \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 2 \
    --eval_steps 5000 \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --train_adapter \
    --adapter_reduction_factor ${adapter_reduction_factor} \
    --language ${lng} \
    --num_train_epochs 6.0 \
    --adapter_config ${adapter_config} \
    --max_train_samples ${data_sample}
