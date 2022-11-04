CKPT=25000
# M="/home/zhengxinyong/outputs/bloom-7b1_ru_continual-pretrain_100000samples_-1vocab_original/"  # 55.87
# M="bigscience/bloom-7b1" # 52.7
M="facebook/xglm-7.5B"
# M="sberbank-ai/mGPT" # 54.6

python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model xglm \
--model_args tokenizer=$M,pretrained=$M \
--device cuda:6 \
--tasks xwinograd_ru \
--no_cache

# --model_args tokenizer="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4",pretrained="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4" \
# --model_args tokenizer="bigscience/bloom-7b1",pretrained="bigscience/bloom-7b1" \