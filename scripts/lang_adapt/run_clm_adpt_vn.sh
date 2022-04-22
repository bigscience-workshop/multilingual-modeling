#!/bin/bash

# Ask for the GPU partition and 1 GPU
#SBATCH -p gpu
#SBATCH --gres="gpu:1"

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=100g

# Specify a job name:
#SBATCH -J run_clm_madx

# Specify an output file
#SBATCH -o /tmp-network/user/vnikouli/Projects/bigscience/logs/run_clm_madx-%j.out
#SBATCH -e /tmp-network/user/vnikouli/Projects/bigscience/logs/run_clm_madx-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vassilina.nikoulina@naverlabs.com
#SBATCH --constraint="gpu_v100&gpu_32g"

FP_BIGS=/tmp-network/user/vnikouli/Projects/bigscience
# Set up the environment by loading modules
source $FP_BIGS/multilingual-modeling/scripts/env/bin/activate

data_sample=$1
ch=118500
lng=$2
adapter_reduction_factor=$3
dataset=oscar
adapter_config="pfeiffer+inv"
vocabsize=24000
model_name="tr5b-1B3-multilingual-alpha-checkpoints/ch${ch}"
tokenizer_dir="${FP_BIGS}/tokenizers/${lng}_oscar_${data_sample}_tokenizer_${vocabsize}" #default tok settings with vocab size = 24k
cache_dir="${FP_BIGS}/data/"
data_dir="${FP_BIGS}/exp-ext-${lng}/madx-bs1b3-multi-ch${ch}-${lng}-sample${data_sample}-$( basename $tokenizer_dir )"
data_tok_dir=${data_dir}/lng_tok

output_dir="${data_dir}/bs1.3B${ch}-${adapter_config}-${adapter_reduction_factor}-es5"
logging_dir="${FP_BIGS}/logs/exp-ext-${lng}/madx-bs1b3-multi-ch${ch}-${lng}-sample${data_sample}-$( basename $tokenizer_dir )/bs1.3B${ch}-${adapter_config}-${adapter_reduction_factor}-es5"
echo $output_dir

BIGS_MODEL=${FP_BIGS}/multilingual-modeling/scripts/exp-009/$model_name 


mkdir -p $output_dir
mkdir -p $logging_dir

adapter_config="pfeiffer+inv"
python $FP_BIGS/multilingual-modeling/scripts/lang_adapt/madx_run_clm.py \
    --seed 0 \
    --fp16 \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_${lng}" \
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
    --eval_steps 1000 \
    --evaluation_strategy "epoch" \
    --max_eval_samples 5000 \
    --save_steps 10000 \
    --save_strategy "steps" \
    --save_total_limit 3 \ 
    --max_train_samples $data_sample \
    --max_steps 50000 \
    --train_adapter \
    --load_best_model_at_end \
    --lang_adapt_strategies "emb-and-adpt" \
    --embedding_strategies "overlap-replace" \
    --adapter_reduction_factor $adapter_reduction_factor \
    --adapter_config ${adapter_config} \
    --language $lng
