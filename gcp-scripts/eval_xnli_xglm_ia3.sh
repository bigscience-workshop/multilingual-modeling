tok="bigscience/bloom-7b1"
M="/home/zhengxinyong/outputs/bloom-7b1_ru_ia3+inv_100000samples_-1vocab_original-frozen/oscar_ia3+inv_ru"

python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer=$tok,pretrained=$tok,adapter_ckpt_folder=$M \
--tasks xnli_ru \
--no_cache \
--device cuda:0