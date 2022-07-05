# Run on JZ

## Getting Started
Clone the GitHub Repository and `cd` into it to run commands like `sbatch jz/emb.sh my 100000 24000 extend`.

```
git clone https://github.com/bigscience-workshop/multilingual-modeling.git
cd multilingual-modeling/
```

## Change Configuration
### SLURM Configuration
We need to change the SLURM setting according to JZ to get the necessary compute.
```
# use a single V100 for each run
#SBATCH --partition=gpu-he --gres=gpu:1  

# output/error files for tracking pip installation
#SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/misc/lang-adapt-env_jz_lang_adapter.out
#SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/misc/lang-adapt-env_jz_lang_adapter.err
```

### Directory configuration (Line 22 - 28 in jz/emb.sh)
Also, we need to change 6 lines of the directory configuration.
```
# virtual environment folder for `python3 -m venv $env_dir`
env_dir="/users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling/jz/env_jz_lang_adapter"

# cache directory for HuggingFace datasets
cache_dir="/users/zyong2/data/zyong2/huggingface"

# cloned GitHub directory
mm_dir="/users/zyong2/data/zyong2/bigscience/gh/multilingual-modeling"

# directory to save adapted models and trained tokenizers
output_dir="/users/zyong2/data/zyong2/bigscience/data/processed/misc/"  

# folder for storing error and output logging text files
logging_txt_dir="/users/zyong2/data/zyong2/bigscience/logs/misc"  

# folder for storing all tensorboard logging
logging_tb_dir="/users/zyong2/data/zyong2/bigscience/reports/misc/"
```

## Runs
### 07/05/2022 (Language Adaptation - Embedding-only)
Run the following commands for doing language adaptation for 4 languages varying along the the size of training samples. 
```
sbatch jz/emb.sh my 100000 24000 extend
sbatch jz/emb.sh my 10000 5000 extend
sbatch jz/emb.sh my 1000 5000 extend

sbatch jz/emb.sh si 100000 24000 extend
sbatch jz/emb.sh si 10000 5000 extend
sbatch jz/emb.sh si 1000 5000 extend

sbatch jz/emb.sh az 100000 24000 extend
sbatch jz/emb.sh az 10000 5000 extend
sbatch jz/emb.sh az 1000 5000 extend

sbatch jz/emb.sh de 100000 24000 extend
sbatch jz/emb.sh de 10000 5000 extend
sbatch jz/emb.sh de 1000 5000 extend
```