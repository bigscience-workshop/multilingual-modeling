LANG="de"
DATA_SAMPLES=100000
VOCAB_SIZE=-1

BIGS_MODEL="bigscience/bloom-1b1"
ADPT_STRATEGY="continual-pretrain"
EMBD_SRATEGY="original"

tokenizer_dir="bigscience/bloom-1b1"
cache_dir="/home/zhengxinyong/.cache/huggingface"
output_dir="/home/zhengxinyong/outputs/$(basename $BIGS_MODEL)_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"
logging_dir="/home/zhengxinyong/logs/$(basename $BIGS_MODEL)_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"

mkdir -p $output_dir
mkdir -p $logging_dir

BSZ=2
deepspeed --num_gpus=4 --master_port 60000 \
    /home/zhengxinyong/multilingual-modeling/scripts/lang_adapt/madx_run_clm.py \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --cache_dir $cache_dir \
    --dataset_name oscar \
    --dataset_config_name "unshuffled_deduplicated_${LANG}" \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 0.0001 \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $BSZ \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 8 \
    --eval_steps 5000 \
    --evaluation_strategy "steps" \
    --max_eval_samples 1000 \
    --save_steps 5000 \
    --save_strategy "steps" \
    --max_train_samples $DATA_SAMPLES \
    --max_steps 25000 \
    --logging_steps 2500 \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --load_best_model_at_end \
    --deepspeed "/home/zhengxinyong/multilingual-modeling/scripts/lang_adapt/ds_config_zero2.json"
