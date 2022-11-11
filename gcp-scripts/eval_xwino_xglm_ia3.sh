tok="bigscience/bloom-7b1"
M="/home/zhengxinyong/outputs/bloom-7b1_ru_ia3+inv_100000samples_-1vocab_original-frozen/oscar_ia3+inv_ru"
# M="bigscience/bloom-7b1" # 52.7
# M="facebook/xglm-7.5B"
# M="sberbank-ai/mGPT" # 54.6

python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer=$tok,pretrained=$tok,adapter_ckpt_folder=$M \
--device cuda:0 \
--tasks xwinograd_ru \
--no_cache