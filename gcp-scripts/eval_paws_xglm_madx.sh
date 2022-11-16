CKPT=25000
B="bigscience/bloom-3b"
M="/home/zhengxinyong/outputs/bloom-3b_de_pfeiffer+inv_100000samples_-1vocab_original-frozen/oscar_pfeiffer+inv_de"

python3 /home/zhengxinyong/lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer=$B,pretrained=$B,adapter_ckpt_folder=$M \
--device cuda:1 \
--tasks pawsx_de \
--no_cache