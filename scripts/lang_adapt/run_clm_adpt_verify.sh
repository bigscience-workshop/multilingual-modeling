# #!/bin/bash

# # Request half an hour of runtime:
# #SBATCH --time=2-23:59:00

# # Ask for the GPU partition and 1 GPU
# #SBATCH --partition=gpu-he --gres=gpu:1
# #SBATCH --array=200,500

# # Default resources are 1 core with 2.8GB of memory.
# #SBATCH --ntasks=4

# # Use more memory (10GB) (CPU RAM):
# #SBATCH --mem=50g

# # Specify a job name:
# #SBATCH -J exp-020-run_clm_adpt

# # Specify an output file
# #SBATCH -o /users/zyong2/data/zyong2/bigscience/logs/log-020/run_clm_adpt_%a.out
# #SBATCH -e /users/zyong2/data/zyong2/bigscience/logs/log-020/run_clm_adpt_%a.err

# # Set up the environment by loading modules
# set -a # automatically export all variables
# source ~/.env
# set +a

# module load python/3.7.4
# module load gitlfs/2.7.1
# source $FP_BIGS/env_lang_adapter/bin/activate


# axis
LANG="de"
MAX_TRAIN_SAMPLES=100_000 #$(($SLURM_ARRAY_TASK_ID * 1000))
BIGS_MODEL="bigscience/bloom-1b3" # "/users/zyong2/data/zyong2/huggingface/bigscience/tr5b-1B3-multilingual-alpha-checkpoints"
ADPT_REDUCTION_FACTOR=16
adapter_config="pfeiffer+inv"

ADPT_STRAT="emb-and-adpt"
EMB_STRAT="overlap-replace"

tokenizer_dir=./trained_tokenizers//tok_bloom-1b3_de_oscar_100000samples_24000vocab_replace
cache_dir=./cache #"/users/zyong2/data/zyong2/huggingface/"
output_dir="./KEEP_RESULTS/de/1b3-50k-prelora-"$adapter_config"-"$MAX_TRAIN_SAMPLES"samples-"$EMB_STRAT"-"$ADPT_STRAT-"$ADPT_REDUCTION_FACTOR"reduction""
logging_dir="./KEEP_RESULTS/de/1b3-50k-prelora-"$adapter_config"-"$MAX_TRAIN_SAMPLES"samples-"$EMB_STRAT"-"$ADPT_STRAT-"$ADPT_REDUCTION_FACTOR"reduction""
mkdir -p $output_dir
mkdir -p $logging_dir

cp ./run_clm_adpt_verify.sh $output_dir/run_clm_adpt.sh

# CUDA_VISIBLE_DEVICES=4 python ../../../dev/multilingual-modeling/scripts/lang_adapt/madx_run_clm.py \
CUDA_VISIBLE_DEVICES=6 python ./madx_run_clm.py \
    --fp16 \
    --seed 0 \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_$LANG" \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 0.001 \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 1 \
    --eval_accumulation_steps 8 \
    --eval_steps 2500 \
    --logging_steps 100 \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --save_steps 12500 \
    --save_strategy "steps" \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --max_steps 50000 \
    --train_adapter \
    --lang_adapt_strategies $ADPT_STRAT \
    --embedding_strategies $EMB_STRAT \
    --adapter_reduction_factor $ADPT_REDUCTION_FACTOR \
    --adapter_config ${adapter_config} \
    --language $LANG \
    --load_best_model_at_end
    # --gradient_checkpointing


# {'loss': 6.4383, 'learning_rate': 0.00099808, 'epoch': 0.02}                                                                                                                                                                                                                                                   
# {'loss': 4.9892, 'learning_rate': 0.0009960799999999999, 'epoch': 0.03}                                                                                                                                                                                                                                        
# {'loss': 4.6186, 'learning_rate': 0.00099408, 'epoch': 0.05}                                                                                                                                                                                                                                                   
# {'loss': 4.47, 'learning_rate': 0.00099208, 'epoch': 0.06}                                                                                                                                                                                                                                                     
# {'loss': 4.348, 'learning_rate': 0.00099008, 'epoch': 0.08}                                                                                                                                                                                                                                                    
# {'loss': 4.2222, 'learning_rate': 0.00098808, 'epoch': 0.09}                                                                                                                                                                                                                                                   
# {'loss': 4.1662, 'learning_rate': 0.00098608, 'epoch': 0.11}                                                                                                                                                                                                                                                   
# {'loss': 4.145, 'learning_rate': 0.00098408, 'epoch': 0.12}                                                                                                                                                                                                                                                    
# {'loss': 4.0381, 'learning_rate': 0.00098208, 'epoch': 0.14}                                                                                                                                                                                                                                                   
# {'loss': 4.0502, 'learning_rate': 0.00098008, 'epoch': 0.15}                                                                                                                                                                                                                                                   
# {'loss': 3.9835, 'learning_rate': 0.0009780799999999999, 'epoch': 0.17}                                                                                                                                                                                                                                        
# {'loss': 3.9635, 'learning_rate': 0.0009760799999999999, 'epoch': 0.18}                                                                                                                                                                                                                                        
# {'loss': 3.9859, 'learning_rate': 0.00097408, 'epoch': 0.2}                                                                                                                                                                                                                                                    
# {'loss': 3.916, 'learning_rate': 0.0009720800000000001, 'epoch': 0.21}                                                                                                                                                                                                                                         
#   3%|███████▎                                                              