language="en-ru"

# model_name="bigscience/bloom-7b1"
model_name="/home/zhengxinyong/outputs/bloom-1b1_ru_continual-pretrain_100000samples_-1vocab_original/checkpoint-5000"

base_model="bigscience/bloom-1b1"

python3 /home/zhengxinyong/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--cache_dir "/home/zhengxinyong/.cache/huggingface" \
--lang_pairs $language \
--model_name $model_name \
--tokenizer $base_model \
--base_model $base_model \
--device cuda:6

model_name="/home/zhengxinyong/outputs/bloom-1b1_ru_continual-pretrain_100000samples_-1vocab_original/checkpoint-10000"
python3 /home/zhengxinyong/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--cache_dir "/home/zhengxinyong/.cache/huggingface" \
--lang_pairs $language \
--model_name $model_name \
--tokenizer $base_model \
--base_model $base_model \
--device cuda:6

model_name="/home/zhengxinyong/outputs/bloom-1b1_ru_continual-pretrain_100000samples_-1vocab_original/checkpoint-15000"
python3 /home/zhengxinyong/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--cache_dir "/home/zhengxinyong/.cache/huggingface" \
--lang_pairs $language \
--model_name $model_name \
--tokenizer $base_model \
--base_model $base_model \
--device cuda:6

model_name="/home/zhengxinyong/outputs/bloom-1b1_ru_continual-pretrain_100000samples_-1vocab_original/checkpoint-20000"
python3 /home/zhengxinyong/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--cache_dir "/home/zhengxinyong/.cache/huggingface" \
--lang_pairs $language \
--model_name $model_name \
--tokenizer $base_model \
--base_model $base_model \
--device cuda:6

model_name="/home/zhengxinyong/outputs/bloom-1b1_ru_continual-pretrain_100000samples_-1vocab_original/checkpoint-25000"
python3 /home/zhengxinyong/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--cache_dir "/home/zhengxinyong/.cache/huggingface" \
--lang_pairs $language \
--model_name $model_name \
--tokenizer $base_model \
--base_model $base_model \
--device cuda:6

model_name="/home/zhengxinyong/outputs/bloom-1b1_ru_continual-pretrain_100000samples_-1vocab_original/"
python3 /home/zhengxinyong/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--cache_dir "/home/zhengxinyong/.cache/huggingface" \
--lang_pairs $language \
--model_name $model_name \
--tokenizer $base_model \
--base_model $base_model \
--device cuda:6