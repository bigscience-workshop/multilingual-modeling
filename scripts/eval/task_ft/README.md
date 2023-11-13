# XNLI (Cross-Lingual and Supervised Setting)

Current scripts are for XNLI (German). 

```
OUTPUT_DIR=... # where you want to save checkpoints at
LANG="de"
CACHE_DIR=... # cache dir for saving/loading HF models and XNLI datasets.
LR=1e-5
MODEL_NAME="ZYONG2/bigscience/tr5b-1B3-multilingual-alpha-checkpoints" # previous version of BLOOM pre-release
TOKENIZER_NAME="ZYONG2/processed/011/oscar-de-tokenizer"

# language adapters checkpoint folder
MADX_LANG_ADAPTER_NAME=".../oscar_de"

# we finetune task adapters for XNLI
FT_STRATEGIES="task_adapters"

mkdir -p $OUTPUT_DIR
python adapters_xnli_de.py \
$OUTPUT_DIR \
--lang $LANG \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $LR \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--pretrained_model $MODEL_NAME \
--tokenizer $TOKENIZER_NAME \
--do_train \
--do_eval_after_train \
--madx_lang_adapter $MADX_LANG_ADAPTER_NAME \
--finetune_strategies $FT_STRATEGIES \
--zero_shot
```

Remove `--zero_shot` for supervised finetuning setting. `scripts_xnli/train_xnli_zero_shot.sh` is the batch script for XNLI training, and `scripts_xnli/run_eval_xnli_zero_shot.sh` is for evaluating trained XNLI task adapters.

### Zero-shot Prompt-based Setting

See branch [`bigscience-lm-adapt`](https://github.com/yongzx/lm-evaluation-harness/tree/bigscience-lm-adapt) of yongzx/lm-evaluation-harness (forked repo).