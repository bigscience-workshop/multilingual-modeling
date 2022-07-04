
model_name=../lang_adapt/KEEP_RESULTS/de/1b3-postlora-prefix_tuning-100_000samples-overlap-replace-emb-and-adpt-16reduction/checkpoint-12500/
ORIGINAL_MODEL="bigscience/bloom-1b3" 
TOKENIZER_DIR=../lang_adapt/trained_tokenizers/tok_bloom-1b3_de_oscar_100000samples_24000vocab_replace
CACHE_DIR="../lang_adapt/cache"

LANG="de"
LR=1e-5
# language adapters checkpoint folder
MADX_LANG_ADAPTER_NAME="../lang_adapt/KEEP_RESULTS/de/1b3-postlora-prefix_tuning-100_000samples-overlap-replace-emb-and-adpt-16reduction/checkpoint-12500/oscar_prefix_tuning_${LANG}"

# we finetune task adapters for XNLI
FT_STRATEGIES="task_adapters"

outdir=$MODEL_DIR/xnli_eval_zero_shot
# evaluate zero-shot training
CUDA_VISIBLE_DEVICES=1 python eval.py \
./baseline-results/1b3-postlora-prefix_tuning-100_000samples-overlap-replace-emb-and-adpt-16reduction/ \
--dataset xnli \
--lang $LANG \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $LR \
--per_device_train_batch_size 32 \
--gradient_accumulation_steps 1 \
--adapted_model_dir $model_name \
--original_model $ORIGINAL_MODEL \
--tokenizer $TOKENIZER_DIR \
--task_layers "task-adapters" \
--madx_lang_adapter $MADX_LANG_ADAPTER_NAME \
--do_train
# --do_predict

# --baseline \
# --use_partial_data \
# --use_partial_train_data 1000 \
# --use_partial_val_data 100 \
# --use_partial_test_data 100 \
