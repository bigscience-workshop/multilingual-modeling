# README

### Tokenizer and Tokenization of Dataset
Run `tokenized4clm_sampled.py` to train the tokenizer on the subset of OSCAR dataset.
- `lang`: language name (e.g., "de", "th")
- `model`: original tokenizer (e.g., "bigscience/bloom-1b3")
- `tokenizer_dir`: path directory to save the tokenizer. The tokenizer will be saved as `tok_${model}_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_{replace/extend}`
- `cache_dir` (default is "~/.cache/huggingface/transformers"): cache directory for downloading the OSCAR dataset and GPT2 tokenizer.
- `vocab_size`: vocab size of the tokenizer
- `sample_size`: the amount of samples to use to train the  tokenizer (randomly selected)
- `use_auth_token`: must be used for BLOOM model
- `extend`: if set, it means that we are extending instead of replacing.

```
tokenizer_dir=... # directory to save trained tokenizer
cache_dir=... # directory to cache downloaded HF model
lang=...  # language
sample_size=...  # training sample size
vocab_size=...  # vocab size of tokenizer
model="bigscience/bloom-1b3"

python ./scripts/lang_adapt/tokenized4clm_sampled.py \
--lang $lang \
--model $model \
--tokenizer_dir  \
--hf_cache_dir $cache_dir \
--vocab_size $vocab_size \
--sample_size $sample_size \
--use_auth_token
--extend  # use "extend" for the embedding strategy of extending vocab.
```
---

### Language Adaptation (6 Combinations)
- use `sbatch run_clm_emb.sh` to perform language adaptation with (emb-only, replace-vocab) strategies. Replace the LANG variable for the desired language (currently is `th`). Currently, the script uses slurm-job-array to control the size of the oscar training corpora. Note: remember to change the SLURM logging output files, `tokenizer_dir`, `cache_dir`, `output_dir`, and `logging_dir` in `run_clm_emb.sh`. 
- use `sbatch run_clm_adpt.sh` to perform language adaptation with (emb-and-adpt, replace-vocab) strategies. Replace the LANG variable for the desired language (currently is `th`). Currently, the script uses slurm-job-array to control the size of the oscar training corpora and `ADPT_REDUCTION_FACTOR` to control the reduction factor of adapter modules. Note: remember to change the SLURM logging output files, `tokenizer_dir`, `cache_dir`, `output_dir`, and `logging_dir` in `run_clm_adpt.sh`. 
    - Hack: after `trainer.save_model()`, manually save the `wte` and `wpe` weights. 
