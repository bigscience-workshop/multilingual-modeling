import torch
import datasets
import transformers
from transformers import AutoTokenizer
from datasets import load_dataset
import pathlib

import argparse
import sys

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
log_level = -1
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()
tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


parser = argparse.ArgumentParser()
parser.add_argument('--lang', type=str, required=True)
parser.add_argument('--tokenizer_dir', type=str, required=True)
parser.add_argument('--hf_cache_dir', default="~/.cache/huggingface/transformers", type=str)
parser.add_argument('--vocab_size', default=130_000, type=int)
parser.add_argument('--extend_vocab', action='store_true')
args = parser.parse_args()
lang = args.lang
if args.extend_vocab:
    assert args.vocab_size < 100_000

raw_datasets = load_dataset(
    "oscar", 
    f"unshuffled_deduplicated_{lang}", 
    cache_dir=args.hf_cache_dir
)

print(f"✅ Loaded raw_datasets OSCAR language {lang}")

def batch_iterator():
    batch_size = 1000
    for i in range(0, len(raw_datasets), batch_size):
        yield raw_datasets['train'][i : i + batch_size]["text"]

tokenizer = AutoTokenizer.from_pretrained("gpt2")
assert tokenizer.is_fast
new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=args.vocab_size)
print("✅ Trained tokenizer with len ", len(new_tokenizer))

new_tokenizer.save_pretrained(f"{args.tokenizer_dir}/{lang}_oscar_tokenizer_{'full' if not args.extend_vocab else str(args.vocab_size)}")
print(f"✅ Saved tokenizer to {args.tokenizer_dir}/{lang}_oscar_tokenizer_{'full' if not args.extend_vocab else str(args.vocab_size)}")