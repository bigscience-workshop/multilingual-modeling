from datasets import load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(str(Path.home() / ".env"))

dataset = load_dataset("paws-x", 'fr', cache_dir=f"{os.getenv('FP_BIGS')}/data/external/paws-x")
print("Done")