# README

### Tokenizer and Tokenization of Dataset
Run `tokenized4clm_sampled.py` to train the tokenizer on the subset of OSCAR dataset.
- `lang`: language name (e.g., "de", "th")
- `model`: original tokenizer (e.g., "bigscience/bloom-1b3")
- `tokenizer_dir`: path directory to save the tokenizer. The tokenizer will be saved as `tok_${model}_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_{replace/extend}`
- `cache_dir` (default is "~/.cache/huggingface/transformers"): cache directory for downloading the OSCAR dataset and GPT2 tokenizer.
- `vocab_size`: vocab size of the tokenizer
- `sample_size`: the amount of samples to use to train the tokenizer (randomly selected)
- `use_auth_token`: must be used for BLOOM model
- `tok_strategy`: extend, replace or overlap-replace

```
cache_dir=...
output_dir=...
lang=...  # language
sample_size=...  # training sample size
vocab_size=...  # vocab size of tokenizer
tok_strategy=...  # extend, replace, overlap-replace
bigs_model="bigscience/bloom-1b3"

tokenizer_dir="${output_dir}/tok_$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_${tok_strategy}"

python ./scripts/lang_adapt/tokenized4clm_sampled.py \
--lang $lang \
--model $bigs_model \
--tokenizer_dir $tokenizer_dir \
--hf_cache_dir $cache_dir \
--vocab_size $vocab_size \
--sample_size $sample_size \
--use_auth_token \
--tok_strategy $tok_strategy
```
---

### Language Adaptation
Run `madx_run_clm.py` to finetune language model on a new language. 
- `LANG`: language name (e.g., "de", "th") on OSCAR
- `DATA_SAMPLES`: training sample size
- `VOCAB_SIZE`: vocab size of the tokenizer
- `BIGS_MODEL`: bigscience model
- `ADPT_STRATEGY`: language adaptation strategy (train only embedding for now: `"emb"`)
- `EMBD_SRATEGY`: embedding strategy. Either `"replace"` (replace the embedding layer entirely), `"overlap-replace"` (replace but initialize seen vocab with pretrained embedding), or `"extend"` (freeze seen vocab embeddings and add trainable embeddings for unseen vocab)
- `TOK_STRATEGY`: tokenization strategy (either `"replace"` (for embedding strategy of "replace" and "overlap-replace") or `"extend"`)
- `tokenizer_dir`: saved tokenizer directory (used in the tokenization script above)
- `cache_dir`: (as above)
- `output_dir`: directory to save adapted model
- `logging_dir`: directory to log loss curves to tensorboard
- `MAX_STEPS`: training steps
- `EVAL_STEPS`: number of training steps between two evaluations
- `SAVE_STEPS`: number of training steps between saving the checkpoints.
```
LANG=... # language
DATA_SAMPLES=... # training sample size
VOCAB_SIZE=... # vocab size of newly trained tokenizer
BIGS_MODEL="bigscience/bloom-1b3"
ADPT_STRATEGY="emb"  # language adaptation strategy (train only embedding for now)
EMBD_SRATEGY=...  # either "replace", "overlap-replace", or "extend"
TOK_STRATEGY=... # either "replace" (for embedding strategy of "replace" and "overlap-replace") or "extend"

tokenizer_dir=... # as above
tokenizer_dir="${tokenizer_dir}/tok_${BIGS_MODEL##*/}_${LANG}_oscar_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${TOK_STRATEGY}"
cache_dir=... # as above

output_dir=... # directory to save adapted model
output_dir="${output_dir}/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"
logging_dir=... # directory to log loss curves to tensorboard
logging_dir="${logging_dir}/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"

mkdir -p $output_dir
mkdir -p $logging_dir

MAX_STEPS=50000
EVAL_STEPS=5000
SAVE_STEPS=5000

python ./scripts/lang_adapt/madx_run_clm.py \
    --seed 0 \
    --fp16 \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_${LANG}" \
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
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --save_steps $SAVE_STEPS \
    --save_strategy "steps" \
    --max_train_samples $DATA_SAMPLES \
    --max_steps $MAX_STEPS \
    --logging_steps 1000 \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --load_best_model_at_end \
    --use_auth_token
```

### IA3 (No Hyperparameters)

```
# axis
LANG="az"
DATA_SAMPLES=$(($SLURM_ARRAY_TASK_ID * 1000))
VOCAB_SIZE=-1
BIGS_MODEL="bigscience/bloom-350m"

# adapters
EMBD_SRATEGY="original-frozen"
ADPT_STRATEGY="ia3"
ADPT_CONFIG="ia3"

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
    --eval_steps 2500 \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --save_steps 2500 \
    --save_strategy "steps" \
    --max_train_samples $DATA_SAMPLES \
    --max_steps 12500 \
    --logging_steps 2500 \
    --train_adapter \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --adapter_config $ADPT_CONFIG \
    --language $LANG \
    --load_best_model_at_end
```