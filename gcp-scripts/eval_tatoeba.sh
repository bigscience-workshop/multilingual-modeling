language="en-ru"

# model_name="bigscience/bloom-7b1"
model_name="/home/zhengxinyong/outputs/bloom-7b1_ru_pfeiffer+inv_100000samples_-1vocab_original-frozen/"

base_model="bigscience/bloom-7b1"

python3 /home/zhengxinyong/multilingual-modeling/scripts/eval/scripts_tatoeba/pilot_baseline.py \
--cache_dir "/home/zhengxinyong/.cache/huggingface" \
--lang_pairs $language \
--model_name $model_name \
--tokenizer $base_model \
--base_model $base_model \
--device cuda:4