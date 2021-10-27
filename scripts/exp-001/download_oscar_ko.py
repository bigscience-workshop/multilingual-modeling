from datasets import load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(str(Path.home() / ".env"))

dataset = load_dataset("oscar", "unshuffled_deduplicated_ko", cache_dir=f"{os.getenv('FP_BIGS')}/data/external/oscar_ko")

from datasets.filesystems import S3FileSystem
s3 = S3FileSystem(key="KEY", secret="VAL")
dataset.save_to_disk('s3://bigscience-add-lang/oscar_ko', fs=s3)
print("Done")