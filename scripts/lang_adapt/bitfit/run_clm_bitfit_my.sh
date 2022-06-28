#!/bin/bash

# axis
LANG="my"
DATA_SAMPLES=100000 #$(($SLURM_ARRAY_TASK_ID * 1000))
VOCAB_SIZE=5000
CH=118500
BIGS_MODEL="bigscience/bloom-350m"
ADPT_STRATEGY="emb-and-adpt"
EMBD_SRATEGY="extend"
FTNE_STRATEGY="bitfit"

tokenizer_dir="checkpoint/tok_bloom-350m_my_oscar_100000samples_24000vocab_extend"
cache_dir="checkpoint/cache/"
output_dir="checkpoint/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${EMBD_SRATEGY}_${FTNE_STRATEGY}_${DATA_SAMPLES}samples"
logging_dir="checkpoint/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${EMBD_SRATEGY}_${FTNE_STRATEGY}_${DATA_SAMPLES}samples"

mkdir -p $output_dir
mkdir -p $logging_dir

python scripts/lang_adapt/madx_run_clm.py \
    --seed 0 \
    --fp16 \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_${LANG}" \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 1e-4 \
    --lr_scheduler_type "constant" \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 1 \
    --eval_steps 1000 \
    --evaluation_strategy "epoch" \
    --max_eval_samples 5000 \
    --save_steps 10000 \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --max_train_samples ${DATA_SAMPLES}\
    --max_steps 50000 \
    --load_best_model_at_end \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --finetuning_strategies $FTNE_STRATEGY \
    --language $LANG &> $output_dir/train.log
