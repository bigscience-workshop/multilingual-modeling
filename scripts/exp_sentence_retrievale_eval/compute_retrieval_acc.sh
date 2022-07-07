#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres="gpu:1"
#SBATCH --mem=200g
#SBATCH --constraint="gpu_v100&gpu_32g"
# Specify an output file
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vassilina.nikoulina@naverlabs.com


source /tmp-network/user/vnikouli/Projects/bigscience/multilingual-modeling/env_bloom/bin/activate
model=$1
dataset=$2
outdir=$model/retrieval_acc-${dataset}
mkdir -p $outdir
python eval_sentence_retrieval.py $outdir --pretrained_model $model --tokenizer $model --dataset $dataset --pooling "max_min"