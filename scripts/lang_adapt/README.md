# README

### Tokenizer and Tokenization of Dataset
First, run `tokenized4clm.py` to train the tokenizer on OSCAR dataset.
- `lang`: language name (e.g., "de", "th")
- `tokenizer_dir`: path directory to save the tokenizer.
- `hf_cache_dir` (default is "~/.cache/huggingface/transformers"): cache directory for downloading the OSCAR dataset and GPT2 tokenizer.
- `vocab_size`: vocab size of the tokenizer
- `extend_vocab`: whether we are extending the vocabulary of the embedding layer (determines the saved name of the tokenizer and for sanity check purpose with vocab_size)
    - if `--extend_vocab`, we save the tokenizer as `{lang}_oscar_tokenizer_{vocab_size}`
    - otherwise, we save the tokenizer as `{lang}_oscar_tokenizer_full`

Then, 
- use `sbatch run_clm_emb.sh` to perform language adaptation with (emb-only, replace-vocab) strategies. Replace the LANG variable for the desired language (currently is `th`). Currently, the script uses slurm-job-array to control the size of the oscar training corpora. Note: remember to change the SLURM logging output files, `tokenizer_dir`, `cache_dir`, `output_dir`, and `logging_dir` in `run_clm_emb.sh`. 
- use `sbatch run_clm_adpt.sh` to perform language adaptation with (emb-and-adpt, replace-vocab) strategies. Replace the LANG variable for the desired language (currently is `th`). Currently, the script uses slurm-job-array to control the size of the oscar training corpora and `ADPT_REDUCTION_FACTOR` to control the reduction factor of adapter modules. Note: remember to change the SLURM logging output files, `tokenizer_dir`, `cache_dir`, `output_dir`, and `logging_dir` in `run_clm_adpt.sh`.