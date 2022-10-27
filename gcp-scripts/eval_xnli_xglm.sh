python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer="/home/zhengxinyong/outputs/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original",pretrained="/home/zhengxinyong/outputs/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original" \
--device cuda:0 \
--tasks xnli_de

# --model_args tokenizer="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4",pretrained="bs-la/bloom-560m_de_continual-pretrain_100000samples_-1vocab_original_bsz4" \
# --model_args tokenizer="bigscience/bloom-7b1",pretrained="bigscience/bloom-7b1" \