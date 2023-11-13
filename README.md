# README

This repository contains code for performing language adaptation of multilingual pretrained large language model BLOOM-{560m,1b1,1b7,3b,7b1} to new unseen languages. Please refer to our ACL 2023 paper [BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting](https://aclanthology.org/2023.acl-long.653/).

Our implementations support the following features:
- finetuning new tokenizers and embedding layers to support new script of unseen languages.
- different embedding stategies where we replace the entire embedding by training from scratch, reinitialize embedding layers but initialize seen vocabulary, or extend the embedding layer to support new tokens. 
- more than 15 language adaptation strategies for pretrained BLOOM model, including continued-pretraining and parameter-efficient finetuning such as BitFit ([Zaken et al., 2021](https://arxiv.org/abs/2106.10199)), (IA)^3 ([Liu et al., 2022](https://arxiv.org/abs/2205.05638)), LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)), MAD-X ([Pfeiffer et al., 2020](https://aclanthology.org/2020.emnlp-main.617/)), composible sparse finetuning ([Ansell et al., 2022](https://github.com/cambridgeltl/composable-sft)), etc.
- different evaluation settings:
    - supervised fine-tuning or cross-lingual transfer: task-finetuning with (English) task adapters on the following tasks: WikiANN (NER tagging), XLSum (abstractive summarization) and XNLI (natural language inference). This is an artefact that is used for preliminary experiments of our BLOOM+1 work.
    - zero-shot prompting on adapted language models, which is carried out on our [BLOOM+1](https://arxiv.org/abs/2212.09535) paper. This is done with forked and modified EleutherAI's lm-eval-harness library. See branch [`bigscience-lm-adapt`](https://github.com/yongzx/lm-evaluation-harness/tree/bigscience-lm-adapt).


## Installation
1. Install the packages from [composable-sft](https://github.com/cambridgeltl/composable-sft). This is used for composable-SFT finetuning.
2. Install the packages from [rational_activations](https://github.com/ml-research/rational_activations). You would need to follow the [Other CUDA/PyTorch] section for installation. This is used for adaptable-adapters. 
3. Install the packages from this repo using `pip install -r requirements.txt`. 

If encounter error with the `import transformer`, uninstall transformers using the command `pip uninstall transformers` and rerun step 3 to reinstall `transformers` supported by `adapter-transformers` library.

## Experimental Setup (Language Adaptation)

### Tokenizer and Tokenization of Dataset
Run `tokenized4clm_sampled.py` to train the tokenizer on the subset of OSCAR dataset.
- `lang`: language name (e.g., "de", "th")
- `model`: original tokenizer (e.g., "bigscience/bloom-1b3")
- `tokenizer_dir`: path directory to save the tokenizer. The tokenizer will be saved as `tok_${model}_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_{replace/extend}`
- `cache_dir` (default is "~/.cache/huggingface/transformers"): cache directory for downloading the OSCAR dataset and GPT2 tokenizer.
- `vocab_size`: vocab size of the tokenizer
- `sample_size`: the amount of samples to use to train the tokenizer (randomly selected)
- `tok_strategy`: extend, replace or overlap-replace

```
cache_dir=...
output_dir=...
lang=...  # language
sample_size=...  # training sample size
vocab_size=...  # vocab size of tokenizer
tok_strategy=...  # extend, replace, overlap-replace
bigs_model="bigscience/bloom-1b3"

tokenizer_dir="${output_dir}/tok_$(basename $bigs_model)_${lang}_oscar_${sample_size}samples_${vocab_size}vocab_${tok_strategy}"

python ./scripts/lang_adapt/tokenized4clm_sampled.py \
--lang $lang \
--model $bigs_model \
--tokenizer_dir $tokenizer_dir \
--hf_cache_dir $cache_dir \
--vocab_size $vocab_size \
--sample_size $sample_size \
--tok_strategy $tok_strategy
```
---

### Language Adaptation
Run `madx_run_clm.py` to finetune language model on a new language. 
- `LANG`: language name (e.g., "de", "th") on OSCAR
- `DATA_SAMPLES`: training sample size
- `VOCAB_SIZE`: vocab size of the tokenizer
- `BIGS_MODEL`: bigscience model
- `ADPT_STRATEGY`: language adaptation strategy 
    - `"emb"`: train only embedding
    - `"continual-pretrain"`: continued pretraining of the entire BLOOM model
    - `"emb-then-adpt"`: train embedding then Pfeiffer adapter later (sequential training)
    - `"pfeiffer"`, `"pfeiffer+inv"`: Pfeiffer adapters in transformers block. ([Houlsby et al., 2019](https://arxiv.org/abs/1902.00751)) Without or with invertible adapters in embedding layer. This is also known as MAD-X ([Pfeiffer et al., 2020](https://aclanthology.org/2020.emnlp-main.617/)). 
    - `"lora"`: LoRA adapters in transformers block ([Hu et al., 2021](https://arxiv.org/abs/2106.09685))
    - `"aa"`: adaptable adapters ([Moosavi et al., 2022](https://arxiv.org/abs/2205.01549))
    - `"ia3"`, `"ia3+inv"`:  (IA)^3 adapters in transformers block. Without or with invertible adapters in embedding layer. ([Liu et al., 2022](https://arxiv.org/abs/2205.05638))
    - `"prefix_tuning"`, `"prefix_tuning_flat"`: Prefix tuning in input space, whether using MLP layers to initialize (without `flat`) or directly initialize tokens (with `flat`) as prefix tokens. ([Li & Liang, 2021](https://arxiv.org/abs/2101.00190))
    - `"prompt-tuning"`: Prompt-tuning in transformer blocks ([Lester et al., 2021](https://arxiv.org/abs/2104.08691))
    - `"sft"`: Composable sparse finetuning. ([Ansell et al., 2022](https://aclanthology.org/2022.acl-long.125/))
    - `"bitfit"`, `"bitfit+inv"`: Finetuning bias layers. Without or with invertible adapters in embedding layer. ([Zaken et al., 2021](https://arxiv.org/abs/2106.10199))
    - `"fish"`: Finetuning FISH masks. ([Sung et al., 2021](https://arxiv.org/abs/2111.09839))
    - `"compacter"`, `"compacterpp"`: Compacter or compacter++ adapters in transformer blocks. ([Mahabadi et al., 2021](https://arxiv.org/abs/2106.04647))
- `EMBD_SRATEGY`: embedding strategy. Either `"replace"` (replace the embedding layer entirely), `"overlap-replace"` (replace but initialize seen vocab with pretrained embedding), or `"extend"` (freeze seen vocab embeddings and add trainable embeddings for unseen vocab)
- `TOK_STRATEGY`: tokenization strategy (either `"replace"` (for embedding strategy of "replace" and "overlap-replace") or `"extend"`)
- `tokenizer_dir`: saved tokenizer directory (used in the tokenization script above)
- `cache_dir`: (as above)
- `output_dir`: directory to save adapted model
- `logging_dir`: directory to log loss curves to tensorboard
- `MAX_STEPS`: training steps
- `EVAL_STEPS`: number of training steps between two evaluations
- `SAVE_STEPS`: number of training steps between saving the checkpoints.
```
LANG=... # language
DATA_SAMPLES=... # training sample size
VOCAB_SIZE=... # vocab size of newly trained tokenizer
BIGS_MODEL="bigscience/bloom-1b3"
ADPT_STRATEGY="emb"  # language adaptation strategy (train only embedding for now)
EMBD_SRATEGY=...  # either "replace", "overlap-replace", or "extend"
TOK_STRATEGY=... # either "replace" (for embedding strategy of "replace" and "overlap-replace") or "extend"

tokenizer_dir=... # as above
tokenizer_dir="${tokenizer_dir}/tok_${BIGS_MODEL##*/}_${LANG}_oscar_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${TOK_STRATEGY}"
cache_dir=... # as above

output_dir=... # directory to save adapted model
output_dir="${output_dir}/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"
logging_dir=... # directory to log loss curves to tensorboard
logging_dir="${logging_dir}/${BIGS_MODEL##*/}_${LANG}_${ADPT_STRATEGY}_${DATA_SAMPLES}samples_${VOCAB_SIZE}vocab_${EMBD_SRATEGY}"

mkdir -p $output_dir
mkdir -p $logging_dir

MAX_STEPS=50000
EVAL_STEPS=5000
SAVE_STEPS=5000

python ./scripts/lang_adapt/madx_run_clm.py \
    --seed 0 \
    --fp16 \
    --model_name_or_path $BIGS_MODEL \
    --tokenizer_name $tokenizer_dir \
    --dataset_name oscar \
    --cache_dir $cache_dir \
    --dataset_config_name "unshuffled_deduplicated_${LANG}" \
    --logging_dir $logging_dir \
    --report_to "tensorboard" \
    --learning_rate 0.001 \
    --do_train \
    --do_eval \
    --output_dir $output_dir \
    --preprocessing_num_workers 8 \
    --overwrite_output_dir \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 2 \
    --eval_accumulation_steps 4 \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy "steps" \
    --max_eval_samples 5000 \
    --save_steps $SAVE_STEPS \
    --save_strategy "steps" \
    --max_train_samples $DATA_SAMPLES \
    --max_steps $MAX_STEPS \
    --logging_steps 1000 \
    --lang_adapt_strategies $ADPT_STRATEGY \
    --embedding_strategies $EMBD_SRATEGY \
    --load_best_model_at_end \
    --gradient_checkpointing \
    --fp16
```

**BLOOM+1 Reproduction**: See `./scripts/lang_adapt/example_scripts/run_clm_ru_madx_560m.sh` to reproduce language adapation of BLOOM-560m models to Russian in our [BLOOM+1 paper](https://arxiv.org/abs/2212.09535).

### Language Adaptation with DeepSpeed
1. Replace `python ./scripts/lang_adapt/madx_run_clm.py` with `deepspeed --num_gpus=8 --master_port 60000`.
2. Pass deepspeed config file argument `--deepspeed "/home/zhengxinyong/multilingual-modeling/scripts/lang_adapt/ds_config_zero2.json" `

See example file at `./scripts/lang_adapt/example_scripts/run_clm_ru_madx_7b1_deepspeed.sh`, which adapts BLOOM-7b1 model on Google Cloud 8 A100 GPUs. 

## Experimental Setup (Evaluation)

### Zero-Shot Prompting

Prompt the adapted language model in a zero-shot fashion without any finetuning. You'll need to `git clone https://github.com/yongzx/lm-evaluation-harness/tree/bigscience-lm-adapt` to be able to run the experiments. 

Here shows the evaluation code for XNLI zero-shot prompting. You can find it in `lm-evaluation-harness/examples/`. 

For BLOOM+1, the tasks used are: 
- `xnli` ([XNLI: Evaluating Cross-lingual Sentence Representations](https://arxiv.org/abs/1809.05053))
- `amnli` ([AmericasNLI: Evaluating Zero-shot Natural Language Understanding of Pretrained Multilingual Models in Truly Low-resource Languages](https://arxiv.org/abs/2104.08726))
- `pawsx` ([PAWS-X: A Cross-lingual Adversarial Dataset for Paraphrase Identification](https://arxiv.org/abs/1908.11828))
- `xcopa` ([XCOPA: A Multilingual Dataset for Causal Commonsense Reasoning](https://arxiv.org/abs/2005.00333))
- `xstory` (Multilingual [Story Cloze Test and ROCStories Corpora](https://cs.rochester.edu/nlp/rocstories/))
- `xwino`([Wino-X: Multilingual Winograd Schemas for Commonsense Reasoning and Coreference Resolution](https://aclanthology.org/2021.emnlp-main.670/))


**Baseline or Model-Based (BitFit, FISH Mask, etc.)**
```
python3 lm-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer="bigscience/bloom-560m",pretrained="ZYONG2/saved_models/bloom-560m_de_bitfit_100000samples_-1vocab_original-frozen" \
--tasks xnli_de
```

**Using Adapters (MAD-X, Pfeiffer, IA3, LoRA, etc.)**
```
python3 m-evaluation-harness/main.py \
--model bigscience \
--model_args tokenizer="bigscience/bloom-560m",pretrained="bigscience/bloom-560m",adapter_ckpt_folder="ZYONG2/saved_models/bloom-560m_de_ia3_100000samples_-1vocab_original-frozen/oscar_ia3_de" \
--tasks xnli_de
```

### Supervised Finetuning or Cross-Lingual Transfer (Only used for preliminary experiments with BLOOM is released)
```
OUTPUT_DIR=... # where you want to save checkpoints at
LANG="de"
CACHE_DIR=... # cache dir for saving/loading HF models and XNLI datasets.
LR=1e-5
MODEL_NAME="ZYONG2/bigscience/tr5b-1B3-multilingual-alpha-checkpoints" # previous version of BLOOM pre-release
TOKENIZER_NAME="ZYONG2/processed/011/oscar-de-tokenizer"

# language adapters checkpoint folder
MADX_LANG_ADAPTER_NAME=".../oscar_de"

# we finetune task adapters for XNLI
FT_STRATEGIES="task_adapters"

mkdir -p $OUTPUT_DIR
python adapters_xnli_de.py \
$OUTPUT_DIR \
--lang $LANG \
--cache_dir $CACHE_DIR \
--num_train_epochs 2 \
--learning_rate $LR \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 4 \
--pretrained_model $MODEL_NAME \
--tokenizer $TOKENIZER_NAME \
--do_train \
--do_eval_after_train \
--madx_lang_adapter $MADX_LANG_ADAPTER_NAME \
--finetune_strategies $FT_STRATEGIES \
--zero_shot
```

Remove `--zero_shot` for supervised finetuning setting. 

See example scripts in `./scripts/eval/task_ftscripts_xnli/`. `train_xnli_zero_shot.sh` is the batch script for XNLI finetuning, and `run_eval_xnli_zero_shot.sh` is for evaluating trained XNLI task adapters.

## Citation
```
@inproceedings{yong-etal-2023-bloom,
    title = "{BLOOM}+1: Adding Language Support to {BLOOM} for Zero-Shot Prompting",
    author = "Yong, Zheng Xin  and Schoelkopf, Hailey  and Muennighoff, Niklas  and Aji, Alham Fikri  and Adelani, David Ifeoluwa  and Almubarak, Khalid  and Bari, M Saiful  and Sutawika, Lintang  and Kasai, Jungo  and Baruwa, Ahmed  and Winata, Genta  and Biderman, Stella  and Raff, Edward  and Radev, Dragomir  and Nikoulina, Vassilina",
    editor = "Rogers, Anna  and Boyd-Graber, Jordan  and Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.653",
    doi = "10.18653/v1/2023.acl-long.653",
    pages = "11682--11703",
}
```