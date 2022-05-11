#!/bin/bash                                                                                                                                                                                                       
#SBATCH -p gpu                                                                                                                                                                                                    
#SBATCH --gres="gpu:1"                                                                                                                                                                                            
#SBATCH --mem=100g                                                                                                                                                                                                

#SBATCH --mail-type=BEGIN,END,FAIL                                                                                                                                                                                 
#SBATCH --mail-user=vassilina.nikoulina@naverlabs.com                                                                                                                                                              
#SBATCH --constraint="gpu_v100&gpu_32g"  

FP_BIGS=/tmp-network/user/vnikouli/Projects/bigscience
# Set up the environment by loading modules                                                                                                                                                                        
source $FP_BIGS/multilingual-modeling/scripts/env/bin/activate

# XNLI (Cross-Lingual and Supervised Setting)

LANG=$1
data_sample=$2
vocabsize=$3
adapter_reduction_factor=$4

ch=118500


adapter_config="pfeiffer+inv"
model_name="tr5b-1B3-multilingual-alpha-checkpoints/ch${ch}"
ORIGINAL_MODEL=${FP_BIGS}/multilingual-modeling/scripts/exp-009/$model_name 
TOKENIZER_DIR="${FP_BIGS}/tokenizers/${LANG}_oscar_${data_sample}_tokenizer_${vocabsize}" #default tok settings with vocab size = 24k
CACHE_DIR="${FP_BIGS}/data/"
data_dir="${FP_BIGS}/exp-ext-${LANG}/madx-bs1b3-multi-ch${ch}-${LANG}-sample${data_sample}-$( basename $TOKENIZER_DIR )"
data_tok_dir=${data_dir}/lng_tok

MODEL_DIR="${data_dir}/bs1.3B${ch}-${adapter_config}-${adapter_reduction_factor}-es5"
XNLI_ZH_DIR=$ORIGINAL_MODEL/xnli_task_adapter_full
LR=1e-5

# language adapters checkpoint folder
MADX_LANG_ADAPTER_NAME="$MODEL_DIR/oscar_${LANG}"

# we finetune task adapters for XNLI
FT_STRATEGIES="task_adapters"

outdir=$MODEL_DIR/xnli_eval_zero_shot
# evaluate zero-shot training
python adapters_xnli_de_vn.py \
$XNLI_ZH_DIR \
--lang $LANG \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $LR \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--pretrained_model $MODEL_DIR \
--original_model $ORIGINAL_MODEL \
--tokenizer $TOKENIZER_DIR \
--do_eval_after_train \
--madx_lang_adapter $MADX_LANG_ADAPTER_NAME \
--finetune_strategies "task_adapters" \
--zero_shot &> $XNLI_ZH_DIR/$( basename $data_dir )-$( basename $MODEL_DIR )_eval.log




#Remove `--zero_shot` for supervised finetuning setting.

### Zero-shot Prompt-based Setting

#See branch [`bigscience-lm-adapt`](https://github.com/yongzx/lm-evaluation-harness/tree/bigscience-lm-adapt) of yongzx/lm-evaluation-harness (forked repo).
