OUTPUT_DIR=./xlsum_ckpts # where to save checkpoints
LANG="thai" # language name, e.g. "thai" not "th" for xlsum. language code e.g. "de" for xnli.
TASK="xlsum" # xlsum or xnli
CACHE_DIR=~/.cache/huggingface/ # cache dir for saving/loading HF models and datasets
LR=1e-5
MODEL_NAME="bigscience/tr5b-1B3-multilingual-alpha-checkpoints"
TOKENIZER_NAME="bigscience/tr5b-1B3-multilingual-alpha-checkpoints"
REVISION="global_step118500" # branch name, e.g. "global_step118500", if applicable

DEEPSPEED_CONFIG="./deepspeed_config.json" # deepspeed config file, if using deepspeed 
# language adapters checkpoint folder
MADX_LANG_ADAPTER_NAME=""

# only finetune task adapters
FT_STRATEGIES="task_adapters"


mkdir -p $OUTPUT_DIR
deepspeed --include localhost:0 adapters_xlsum_de.py \
$OUTPUT_DIR \
--lang $LANG \
--dataset $TASK \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $LR \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--pretrained_model $MODEL_NAME \
--tokenizer $TOKENIZER_NAME \
--do_train \
--do_eval_after_train \
--use_partial_data \
--zero_shot \
--revision "$REVISION" \
--adapter_lang_name "xlsum-de" \
--finetune_strategies $FT_STRATEGIES \
# --use_partial_data 
# --deepspeed $DEEPSPEED_CONFIG

# --madx_lang_adapter $MADX_LANG_ADAPTER_NAME \