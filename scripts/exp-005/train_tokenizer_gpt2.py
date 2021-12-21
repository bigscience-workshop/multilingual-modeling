from datasets import load_dataset

import os
from pathlib import Path


lang = "de"
dataset = load_dataset("oscar", f"unshuffled_deduplicated_{lang}", cache_dir=f"/users/zyong2/data/zyong2/bigscience/data/external/oscar_{lang}")

def batch_iterator():
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        yield dataset['train'][i : i + batch_size]["text"]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
assert tokenizer.is_fast
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=50_257)
new_tokenizer.save_pretrained(f"/users/zyong2/data/zyong2/bigscience/data/processed/exp-005/oscar-{lang}-tokenizer")