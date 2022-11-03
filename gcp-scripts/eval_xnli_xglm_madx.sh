tok="bigscience/bloom-1b1"
CKPT=25000
M="/home/zhengxinyong/outputs/bloom-1b1_ru_pfeiffer+inv_100000samples_-1vocab_original-frozen/oscar_pfeiffer+inv_ru"

python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer=$tok,pretrained=$tok,adapter_ckpt_folder=$M \
--tasks xnli_ru \
--no_cache \
--device cuda:0