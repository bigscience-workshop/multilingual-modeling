#!/bin/bash

# Request half an hour of runtime:
#SBATCH --time=2-23:59:00

# Ask for the GPU partition and 1 GPU
#SBATCH --partition=gpu-he --gres=gpu:1

# Default resources are 1 core with 2.8GB of memory.
#SBATCH --ntasks=8

# Use more memory (10GB) (CPU RAM):
#SBATCH --mem=200g

# Specify a job name:
#SBATCH -J lang-adapt-env_jz_lang_adapter

# Specify an output file
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/misc/lang-adapt-env_jz_lang_adapter.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/misc/lang-adapt-env_jz_lang_adapter.err

env_dir="/users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/jz/env_jz_lang_adapter"
cache_dir="/users/zyong2/data/zyong2/huggingface"
mm_dir="/users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling"

output_dir="/users/zyong2/data/zyong2/bigscience/data/processed/misc/"  # adapted model and trained tokenizer directory
logging_txt_dir="/users/zyong2/data/zyong2/bigscience/logs/misc"  # error and output logging
logging_tb_dir="/users/zyong2/data/zyong2/bigscience/reports/misc/"  # tensorboard logging

mkdir -p $output_dir
mkdir -p $logging_tb_dir
mkdir -p $logging_txt_dir

lang=$1  # language
sample_size=$2  # training sample size
vocab_size=$3  # vocab size of tokenizer
tok_strategy=$4  # extend, replace, overlap-replace
bigs_model="bigscience/bloom-1b3"
adpt_strategy="emb"

tokenizer_dir="${output_dir}/tok_$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_${tok_strategy}"
logging_tb_dir="${logging_tb_dir}/$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_tok-${tok_strategy}_adpt-${adpt_strategy}"

# setup environment
module load python/3.7.4
[ -d $env_dir ] || python3 -m venv $env_dir
source "${env_dir}/bin/activate"
# pip3 install --upgrade pip
# pip3 install -r "${mm_dir}/requirements.txt"

# train tokenizer
python "${mm_dir}/scripts/lang_adapt/tokenized4clm_sampled.py" \
--lang $lang \
--model $bigs_model \
--tokenizer_dir $tokenizer_dir \
--hf_cache_dir $cache_dir \
--vocab_size $vocab_size \
--sample_size $sample_size \
--use_auth_token \
--tok_strategy $tok_strategy \
> "${logging_txt_dir}/tok_$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_${tok_strategy}.txt" \
2> "${logging_txt_dir}/tok_$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_${tok_strategy}.err"


# finetune language model for langauge adaptation
python "${mm_dir}/scripts/lang_adapt/madx_run_clm.py" \
    --seed 0 \
    --fp16 \
    --model_name_or_path $bigs_model \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_${lang}" \
    --logging_dir $logging_tb_dir \
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
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --save_steps 5000 \
    --save_strategy "steps" \
    --max_train_samples $sample_size \
    --max_steps 50000 \
    --logging_steps 1000 \
    --lang_adapt_strategies $adpt_strategy \
    --embedding_strategies $tok_strategy \
    --load_best_model_at_end \
    --use_auth_token \
    > "${logging_txt_dir}/$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_tok-${tok_strategy}_adpt-${adpt_strategy}.txt" \
    2> "${logging_txt_dir}/$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_tok-${tok_strategy}_adpt-${adpt_strategy}.err"
