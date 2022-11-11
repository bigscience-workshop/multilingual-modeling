CKPT=25000
M="/home/zhengxinyong/outputs/bloom-7b1_de_continual-pretrain_100000samples_-1vocab_original/"
# M="bigscience/bloom-7b1"
# M="facebook/xglm-4.5B"
# M="sberbank-ai/mGPT"

python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer=$M,pretrained=$M \
--device cuda:0 \
--tasks pawsx_de \
--no_cache

# --model_args tokenizer="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4",pretrained="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4" \
# --model_args tokenizer="bigscience/bloom-7b1",pretrained="bigscience/bloom-7b1" \