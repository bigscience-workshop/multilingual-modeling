from datasets import load_dataset
import os
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('lang', type=str, help='language subset')
args = parser.parse_args()

dataset = load_dataset("oscar", f"unshuffled_deduplicated_{args.lang}", cache_dir=f"/users/zyong2/data/zyong2/bigscience/data/external/oscar_{args.lang}")
print("Done")