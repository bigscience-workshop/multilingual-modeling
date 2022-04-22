# XNLI Evaluation

Use `xnli_v2.py` to run the evaluation on XNLI. 

### With Language Adapters
```
LANG="th"
CACHE_DIR="/users/zyong2/data/zyong2/huggingface/"
lr=5e-5

# Original BigScience model and language-specific tokenizer
MODEL_NAME="/users/zyong2/data/zyong2/huggingface/bigscience/tr5b-1B3-ckpt118500"
TOKENIZER_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/020/th_oscar_tokenizer_24000"

# saved language adapters
MADX_LANG_ADAPTER_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/020/th_adpt_100000samples/oscar_th"

# saved embedding layers
WTE="/users/zyong2/data/zyong2/bigscience/data/processed/020/th_adpt_100000samples/transformer.wte.weight.pt"
WPE="/users/zyong2/data/zyong2/bigscience/data/processed/020/th_adpt_100000samples/transformer.wpe.weight.pt"

# output directory
OUTPUT_DIR="$FP_BIGS/data/processed/021/xnli_th_adpt_100000samples"

mkdir -p $OUTPUT_DIR

# remove --zero_shot for supervised finetuning setting; otherwise, it will be cross-lingual finetuning setting.
# use --use_partial_data to test the code

python xnli_v2.py \
$OUTPUT_DIR \
--lang $LANG \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $lr \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--pretrained_model $MODEL_NAME \
--tokenizer $TOKENIZER_NAME \
--do_train \
--do_eval_after_train \
--madx_lang_adapter $MADX_LANG_ADAPTER_NAME \
--wte $WTE \
--wpe $WPE \
--zero_shot
```

### Embedding only approach (No Language Adapters)
```
LANG="th"
CACHE_DIR="/users/zyong2/data/zyong2/huggingface/"
lr=5e-5

# Saved finetuned model and language-specific tokenizer
MODEL_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/020/th_emb_100000samples"
TOKENIZER_NAME="/users/zyong2/data/zyong2/bigscience/data/processed/020/th_oscar_tokenizer_24000"

# output directory
OUTPUT_DIR="$FP_BIGS/data/processed/021/xnli_th_adpt_100000samples"

mkdir -p $OUTPUT_DIR

# remove --zero_shot for supervised finetuning setting; otherwise, it will be cross-lingual finetuning setting.
# use --use_partial_data to test the code

python xnli_v2.py \
$OUTPUT_DIR \
--lang $LANG \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $lr \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--pretrained_model $MODEL_NAME \
--tokenizer $TOKENIZER_NAME \
--do_train \
--do_eval_after_train \
--zero_shot \
--use_partial_data
```
