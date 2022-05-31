# README

### Tokenizer and Tokenization of Dataset
Run `tokenized4clm.py` to train the tokenizer on OSCAR dataset.
- `lang`: language name (e.g., "de", "th")
- `model`: model that uses this tokenizer (e.g., "gpt2", "bigscience/bloom-1b3`)
- `tokenizer_dir`: path directory to save the tokenizer. The tokenizer will be saved as `{lang}_oscar_tokenizer_{vocab_size}`
- `hf_cache_dir` (default is "~/.cache/huggingface/transformers"): cache directory for downloading the OSCAR dataset and GPT2 tokenizer.
- `vocab_size`: vocab size of the tokenizer


Run `tokenized4clm_sampled.py` to train the tokenizer on the subset of OSCAR dataset.
- `lang`: language name (e.g., "de", "th")
- `tokenizer_dir`: path directory to save the tokenizer. The tokenizer will be saved as `{lang}_oscar_tokenizer_{vocab_size}`
- `hf_cache_dir` (default is "~/.cache/huggingface/transformers"): cache directory for downloading the OSCAR dataset and GPT2 tokenizer.
- `vocab_size`: vocab size of the tokenizer
- `sample_size`: the amount of samples to use to train the  tokenizer (randomly selected)

---

### Language Adaptation (6 Combinations)
- use `sbatch run_clm_emb.sh` to perform language adaptation with (emb-only, replace-vocab) strategies. Replace the LANG variable for the desired language (currently is `th`). Currently, the script uses slurm-job-array to control the size of the oscar training corpora. Note: remember to change the SLURM logging output files, `tokenizer_dir`, `cache_dir`, `output_dir`, and `logging_dir` in `run_clm_emb.sh`. 
- use `sbatch run_clm_adpt.sh` to perform language adaptation with (emb-and-adpt, replace-vocab) strategies. Replace the LANG variable for the desired language (currently is `th`). Currently, the script uses slurm-job-array to control the size of the oscar training corpora and `ADPT_REDUCTION_FACTOR` to control the reduction factor of adapter modules. Note: remember to change the SLURM logging output files, `tokenizer_dir`, `cache_dir`, `output_dir`, and `logging_dir` in `run_clm_adpt.sh`. 
    - Hack: after `trainer.save_model()`, manually save the `wte` and `wpe` weights. 