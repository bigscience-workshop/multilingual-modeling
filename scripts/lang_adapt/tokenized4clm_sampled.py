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
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--tokenizer_dir', type=str, required=True)
parser.add_argument('--hf_cache_dir', default="~/.cache/huggingface/transformers", type=str)
parser.add_argument('--vocab_size', default=130_000, type=int)
parser.add_argument('--extend_vocab', action='store_true')
# parser.add_argument('--replace_with_overlap', action='store_true')
parser.add_argument('--sample_size', default=None, type=int)

args = parser.parse_args()
lang = args.lang
if args.extend_vocab:
    assert args.vocab_size < 100_000

if  args.sample_size:
    raw_datasets = load_dataset(
        "oscar", 
        f"unshuffled_deduplicated_{lang}", 
        cache_dir=args.hf_cache_dir
    )["train"].shuffle(seed=42).select(range(args.sample_size))
   
else: 
    raw_datasets = load_dataset(
        "oscar", 
        f"unshuffled_deduplicated_{lang}", 
        cache_dir=args.hf_cache_dir
    )["train"]
 
print(f"✅ Loaded raw_datasets OSCAR language {lang}")

def batch_iterator():
    global unique_toks
    batch_size = 1000
    for i in range(0, len(raw_datasets), batch_size):
        sample = raw_datasets[i : i + batch_size]["text"]
        unique_toks = unique_toks.union(set(" ".join(sample).split(" ")))
        yield sample

unique_toks = set()

if args.extend_vocab:
    tokenizer = AutoTokenizer.from_pretrained('/tmp-network/user/vnikouli/Projects/bigscience/multilingual-modeling/scripts/exp-009/tr5b-1B3-multilingual-alpha-checkpoints/')
    assert tokenizer.is_fast
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=args.vocab_size)
    print("✅ Trained tokenizer with len ", len(new_tokenizer))
    added = tokenizer.add_tokens([tok for tok in new_tokenizer.vocab.keys()])
    print(f"Overlap with previous vocab: {args.vocab_size - added}")
    tokenizer.save_pretrained(f"{args.tokenizer_dir}/{lang}_oscar_{args.sample_size}_tokenizer_{args.vocab_size}_extend")
    print(f"Saved tokenizer to {args.tokenizer_dir}/{lang}_oscar_{args.sample_size}_tokenizer_{args.vocab_size}_extend")

# elif args.replace_with_overlap:
#     # 
#     tokenizer = AutoTokenizer.from_pretrained('/tmp-network/user/vnikouli/Projects/bigscience/multilingual-modeling/scripts/exp-009/tr5b-1B3-multilingual-alpha-checkpoints/', unk_token="<unk>")

#     assert tokenizer.is_fast
#     new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=args.vocab_size)
#     print("✅ Trained tokenizer with len ", len(new_tokenizer))
#     new_tokenizer.save_pretrained(f"{args.tokenizer_dir}/{lang}_oscar_{args.sample_size}_tokenizer_{args.vocab_size}_overlap")
#     print(f"Saved tokenizer to {args.tokenizer_dir}/{lang}_oscar_{args.sample_size}_tokenizer_{args.vocab_size}_overlap")

else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert tokenizer.is_fast
    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=args.vocab_size)
    print("Unique toks, ", len(unique_toks))
    print("✅ Trained tokenizer with len ", len(new_tokenizer))
    new_tokenizer.save_pretrained(f"{args.tokenizer_dir}/{lang}_oscar_{args.sample_size}_tokenizer_{args.vocab_size}_replace")
    print(f"Saved tokenizer to {args.tokenizer_dir}/{lang}_oscar_{args.sample_size}_tokenizer_{args.vocab_size}_replace")
