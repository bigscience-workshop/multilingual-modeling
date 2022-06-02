#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres="gpu:1"
#SBATCH --ntasks=16
#SBATCH --mem=50g

# Specify a job name:
#SBATCH -J eval_retrieval_acc

# Specify an output file
#SBATCH -o /tmp-network/user/vnikouli/Projects/bigscience/logs/eval_retrieval_acc-%j.out
#SBATCH -e /tmp-network/user/vnikouli/Projects/bigscience/logs/eval_retrieval_acc-%j.err

#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vassilina.nikoulina@naverlabs.com


model=$1
dataset=$2
outdir=../exp-009/retrieval_acc_${model}-${dataset}
mkdir $outdir
python eval_sentence_retrieval.py $outdir --pretrained_model $model --tokenizer $model --dataset $dataset
