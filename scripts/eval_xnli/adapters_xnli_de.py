import logging
import argparse
import os

from datasets import load_dataset
from datasets import load_metric
from collections import namedtuple

import torch
import numpy as np
from transformers import TrainingArguments, Trainer, AdapterTrainer
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2ForSequenceClassification, AutoModelForCausalLM

# setup logging
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format="{level} {level.icon} | [{time}] - {message}")


# parser
parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--lang", type=str, default="de")
parser.add_argument("--cache_dir")
parser.add_argument("--num_train_epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--pretrained_model") 
parser.add_argument("--original_model")  
parser.add_argument("--tokenizer")
parser.add_argument("--do_train", default=False, action="store_true")
parser.add_argument("--do_eval_after_train", default=False, action="store_true")
parser.add_argument("--do_predict", default=False, action="store_true")
parser.add_argument("--use_partial_data", default=False, action="store_true")
parser.add_argument("--zero_shot", default=False, action="store_true")

finetune_strategies = ["whole", "lang_adapters", "task_adapters"]
parser.add_argument("--madx_lang_adapter")
parser.add_argument("--adapter_lang_name", required=True)
parser.add_argument("--finetune_strategies", choices=finetune_strategies, required=True)

args = parser.parse_args()
if args.do_eval_after_train:
    args.do_predict = True

if args.original_model is None:
    # here: because the wpe is not saved, pretrained_model is the original bigsciece model
    args.original_model = args.pretrained_model

print("Arguments: ========")
print(args)


# load dataset
if args.zero_shot:
    print("0️⃣ 0-Shot")
    # 0-shot: use english as train and validation
    xnli_en_dataset = load_dataset("xnli", "en", cache_dir=args.cache_dir)
    xnli_dataset = load_dataset("xnli", args.lang, cache_dir=args.cache_dir)
    assert args.lang != "en"

    train_dataset = xnli_en_dataset['train']
    val_dataset = xnli_en_dataset['validation']
    test_dataset = xnli_dataset['test']
else:
    print("👀 Supervised Training")
    xnli_dataset = load_dataset("xnli", args.lang, cache_dir=args.cache_dir)

    train_dataset = xnli_dataset['train']
    val_dataset = xnli_dataset['validation']
    test_dataset = xnli_dataset['test']


# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token  # tokenizer.encode(tokenizer.eos_token) = [0]
if args.zero_shot:
    en_tokenizer = AutoTokenizer.from_pretrained(args.original_model, cache_dir=args.cache_dir) # has to use AutoTokenizer instead of GPT2Tokenizer
    en_tokenizer.pad_token = en_tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

def en_tokenize_function(examples):
    return en_tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)


logger.info("Tokenizing the dataset...")
if args.zero_shot:
    full_train_dataset = train_dataset.map(en_tokenize_function, batched=False)
    full_val_dataset = val_dataset.map(en_tokenize_function, batched=False)
else:
    full_train_dataset = train_dataset.map(tokenize_function, batched=False)
    full_val_dataset = val_dataset.map(tokenize_function, batched=False)

full_test_dataset = test_dataset.map(tokenize_function, batched=False)
small_train_dataset = full_train_dataset.shuffle(seed=42).select(range(100))
small_val_dataset = full_val_dataset.shuffle(seed=42).select(range(100))
small_test_dataset = full_test_dataset.shuffle(seed=42).select(range(100))

logger.info(full_train_dataset[0])
logger.info(full_train_dataset[100])

from datasets import load_metric
metric = load_metric("xnli")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments(
    args.output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_steps=500 if not args.use_partial_data else 10,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=500,
    report_to="tensorboard",
    logging_dir=f"{args.output_dir}/logs",
    load_best_model_at_end=True,
)

def load_model(args, inference=False):

    # FIXME: if we load with GPT2ForSequenceClassification, the embeddings are the original one 
    # even when we call load_adapter
    if args.zero_shot and not inference:
        model = GPT2ForSequenceClassification.from_pretrained(args.pretrained_model, 
                                                            num_labels=3,
                                                            pad_token_id=en_tokenizer.pad_token_id,
                                                            cache_dir=args.cache_dir)
    else:
        model = GPT2ForSequenceClassification.from_pretrained(args.pretrained_model, 
                                                            num_labels=3,
                                                            pad_token_id=tokenizer.pad_token_id,
                                                            cache_dir=args.cache_dir)

    if not args.zero_shot or (args.zero_shot and inference):
        # if not zero shot, that means that we need to replace the embedding layers during training
        # we also need to replace embedding layers during inference
        causal_lm_model = AutoModelForCausalLM.from_pretrained(args.original_model)

        # change the embedding layer of the original big science model
        # by loading the adapters (which has saved lm_head)
        causal_lm_model.resize_token_embeddings(len(tokenizer))
        if args.madx_lang_adapter:
            causal_lm_model.load_adapter(args.madx_lang_adapter, config="pfeiffer+inv")                                    
        
        # model has original bigscience embedding so replace it.
        model.resize_token_embeddings(len(tokenizer))
        model._modules['transformer']._modules['wte'] = causal_lm_model._modules['transformer']._modules['wte']

    if not inference:
        if not args.zero_shot:
            if args.madx_lang_adapter:
                adapter_name = model.load_adapter(args.madx_lang_adapter,
                                                config="pfeiffer+inv",
                                                load_as=args.adapter_lang_name)
        if args.finetune_strategies == "whole":
            model.set_active_adapters(adapter_name)
        elif args.finetune_strategies == "lang_adapters":
            model.train_adapter([args.adapter_lang_name])
        elif args.finetune_strategies == "task_adapters":
            model.add_adapter("xnli-task-adapter")
            model.train_adapter("xnli-task-adapter")
        else:
            raise ValueError("Lack configuration")
        
        print("🔥 ==================== Training: ==================== 🔥")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"🥶 Frozen layer '{name}'")
            else:
                print(f"🚀 Trainable layer '{name}'")
        print(model)
    else:
        print("🔥 ==================== Inference: ==================== 🔥")
        if args.finetune_strategies == "lang_adapters":
            assert args.pretrained_adapters_dir 
            adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/{args.adapter_lang_name}")
            model.set_active_adapters(adapter_name)
        elif args.finetune_strategies == "task_adapters":
            if args.madx_lang_adapter:
                assert args.pretrained_adapters_dir 
                adapter_name = model.load_adapter(args.madx_lang_adapter)
                model.set_active_adapters(adapter_name)
                adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/xnli-task-adapter")
                model.set_active_adapters(adapter_name)
            else:
                # adapter_name = model.load_adapter("/users/zyong2/data/zyong2/bigscience/data/processed/013/xnli_de_de_100K_adpt_16_0shot/checkpoint-24544/xnli-task-adapter")

                # for TGT -> TGT supervised finetuning setting, change adapter_name
                adapter_name = model.load_adapter("/users/zyong2/data/zyong2/bigscience/data/processed/exp-013/task_xnli_de_ft_100000_ori/checkpoint-24544/xnli-task-adapter")
                model.set_active_adapters(adapter_name)
        print(model)

    return model

if args.do_train:
    logger.info("Start Training")
    model = load_model(args)
    trainer = AdapterTrainer(
        model=model, 
        args=training_args, 
        train_dataset=small_train_dataset if args.use_partial_data else full_train_dataset, 
        eval_dataset=small_val_dataset if args.use_partial_data else full_val_dataset, 
        compute_metrics=compute_metrics
    )

    trainer.train()

if args.do_predict:
    if args.do_eval_after_train:
        evaluation_dirs = list(sorted([
                    checkpoint_dir
                    for checkpoint_dir in os.listdir(args.output_dir)
                    if checkpoint_dir.startswith('checkpoint-')
                ], key=lambda x: int(x[len('checkpoint-'):])))
        if args.madx_lang_adapter:
            args.pretrained_adapters_dir = f"{args.output_dir}/{evaluation_dirs[-1]}"
            logger.info(f"[Evaluation] Loading trained model from {evaluation_dirs[-1]}")

    model = load_model(args, inference=True)
    training_args.report_to = list()
    
    trainer = AdapterTrainer(
        model=model, 
        args=training_args, 
        eval_dataset=small_test_dataset if args.use_partial_data else full_test_dataset, 
        compute_metrics=compute_metrics
    )

    print("Evaluate on Test:", trainer.evaluate())