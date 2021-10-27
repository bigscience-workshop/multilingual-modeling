from datasets import load_dataset
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv(str(Path.home() / ".env"))

dataset = load_dataset("oscar", "unshuffled_deduplicated_fr", cache_dir=f"{os.getenv('FP_BIGS')}/data/external/oscar_fr")
print("Done")