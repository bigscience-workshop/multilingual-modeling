# README

- use `download_oscar_fr.sh` to download the datasets. To download datasets for other languages, make the necessary changes on line 8 in the `download_oscar_fr.py`.
- run `train_tokenizer_gpt2.py` to train the tokenizer for the new dataset. Make necessary changes on line 8 to load the dataset and line 20 to save the trained tokenizer.
- run `run_clm.sh` to train GPT-2. Important changes to arguments that might be made:
 - `tokenizer_dir`: directory of saved tokenizer.
 - `cache_dir`: directory of cached dataset from `download_oscar_fr.sh` (remember to make changes to the dataset use in the argument `dataset_name` and `dataset_config_name`).
 - `output_dir`: directory where the gpt2 is checkpointed during training.
 - `ckpt_dir`: used for continuing training from checkpoint.

---

# Decisions

**Dataset**: HF's OSCAR unshuffled_deduplicated_fr 

**Tokenizer**: byte-level Byte-pair encoding tokenizer (same as GPT-2). Training is identical to the section "Using an existing tokenizer" in huggingface's tokenizer_training [tutorial](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)
tokenizer_name: `/users/zyong2/data/zyong2/bigscience/data/processed/exp-001/oscar-fr-tokenizer`
- train the GPT-2 tokenizer with the exact same algorithms and parameters as an existing one.
- vocab_size: 50,257 (same as original GPT-2)

 
