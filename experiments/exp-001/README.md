# Decisions

**Dataset**: HF's OSCAR unshuffled_deduplicated_fr 

**Tokenizer**: byte-level Byte-pair encoding tokenizer (same as GPT-2). Training is identical to the section "Using an existing tokenizer" in huggingface's tokenizer_training [tutorial](https://github.com/huggingface/notebooks/blob/master/examples/tokenizer_training.ipynb)
tokenizer_name: `/users/zyong2/data/zyong2/bigscience/data/processed/exp-001/oscar-fr-tokenizer`
- train the GPT-2 tokenizer with the exact same algorithms and parameters as an existing one.
- vocab_size: 50,257 (same as original GPT-2)


 
