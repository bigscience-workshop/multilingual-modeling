from datasets import load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(str(Path.home() / ".env"))

dataset = load_dataset("oscar", "unshuffled_deduplicated_ko", cache_dir=f"{os.getenv('FP_BIGS')}/data/external/oscar_ko")

def batch_iterator():
    batch_size = 1000
    for i in range(0, len(dataset), batch_size):
        yield dataset['train'][i : i + batch_size]["text"]

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
assert tokenizer.is_fast
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=50_257)
new_tokenizer.save_pretrained(f"{os.getenv('FP_BIGS')}/data/processed/exp-001/oscar-ko-tokenizer")