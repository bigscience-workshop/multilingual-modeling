import logging
import argparse
import os

from datasets import load_dataset
from datasets import load_metric
from collections import namedtuple

import torch
import numpy as np
from transformers import TrainingArguments, Trainer, AdapterTrainer
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# setup logging
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format="{level} {level.icon} | [{time}] - {message}")


KLUE = namedtuple("KLUE", ["klue_split", "num_labels", "metric", "model_type"])
KLUE_TASKS = {
    "topic-cls": KLUE(klue_split="ynat", num_labels=7, metric="f1/macro", model_type="seq-cls"),
    "sts-pearsonr": KLUE(klue_split="sts", num_labels=1, metric="pearsonr", model_type="seq-cls"),
    "sts-binary": KLUE(klue_split="sts", num_labels=1, metric="f1/macro", model_type="seq-cls"),
    "nli": KLUE(klue_split="nli", num_labels=3, metric="accuracy", model_type="seq-cls"),
}

# parser
parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--klue_task", choices=KLUE_TASKS.keys(), default="nli")
parser.add_argument("--lang", type=str, default="ko")
parser.add_argument("--cache_dir")
parser.add_argument("--num_train_epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--pretrained_model")
parser.add_argument("--tokenizer")
parser.add_argument("--do_train", default=False, action="store_true")
parser.add_argument("--do_eval_after_train", default=False, action="store_true")
parser.add_argument("--do_predict", default=False, action="store_true")
parser.add_argument("--use_partial_data", default=False, action="store_true")
parser.add_argument("--zero_shot", default=False, action="store_true")

finetune_strategies = ["whole", "lang_adapters", "task_adapters"]
parser.add_argument("--madx_lang_adapter", required=True)
parser.add_argument("--adapter_lang_name", required=True)
parser.add_argument("--finetune_strategies", choices=finetune_strategies, required=True)
args = parser.parse_args()
if args.do_eval_after_train:
    args.do_predict = True

print("Arguments: ========")
print(args)

# load dataset
klue_dataset = load_dataset("klue", KLUE_TASKS[args.klue_task].klue_split, cache_dir=args.cache_dir)
if args.zero_shot:
    print("0️⃣ 0-Shot")
    xnli_en_dataset = load_dataset("xnli", "en", cache_dir=args.cache_dir)

    if "test" not in klue_dataset:
        _train_dataset = klue_dataset['train'].train_test_split(train_size=0.8, shuffle=True, seed=42)
        train_dataset = xnli_en_dataset['train']
        val_dataset = xnli_en_dataset['validation']
        test_dataset = klue_dataset['validation']
    else:
        train_dataset = xnli_en_dataset['train']
        val_dataset = xnli_en_dataset['validation']
        test_dataset = klue_dataset['test']
else:
    print("👀 Supervised Training")
    if "test" not in klue_dataset:
        _train_dataset = klue_dataset['train'].train_test_split(train_size=0.8, shuffle=True, seed=42)
        train_dataset = _train_dataset['train']
        val_dataset = _train_dataset['test']
        test_dataset = klue_dataset['validation']
    else:
        train_dataset = klue_dataset['train']
        val_dataset = klue_dataset['validation']
        test_dataset = klue_dataset['test']


# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)

def tokenize_function(examples):
    if KLUE_TASKS[args.klue_task].klue_split == "ynat":
        return tokenizer(examples["title"], max_length=128, padding="max_length", truncation=True)
    elif KLUE_TASKS[args.klue_task].klue_split == "sts":
        return tokenizer(f'{examples["sentence1"]} {tokenizer.eos_token} {examples["sentence2"]}', max_length=128, padding="max_length", truncation=True)
    elif KLUE_TASKS[args.klue_task].klue_split == "nli":
        return tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

def postprocessing(example):
    if KLUE_TASKS[args.klue_task].klue_split == "sts":
        example['labels'] = example['labels']['real-label']
        return example
    else:
        return example

logger.info("Tokenizing the dataset...")
tokenizer.pad_token = tokenizer.eos_token  # tokenizer.encode(tokenizer.eos_token) = [0]
full_train_dataset = train_dataset.map(tokenize_function, batched=False).map(postprocessing)
full_val_dataset = val_dataset.map(tokenize_function, batched=False).map(postprocessing)
full_test_dataset = test_dataset.map(tokenize_function, batched=False).map(postprocessing)
small_train_dataset = full_train_dataset.shuffle(seed=42).select(range(100))
small_val_dataset = full_val_dataset.shuffle(seed=42).select(range(100))
small_test_dataset = full_test_dataset.shuffle(seed=42).select(range(100))

logger.info(full_train_dataset[0])
logger.info(full_train_dataset[100])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    if "pearsonr" in KLUE_TASKS[args.klue_task].metric:
        predictions = logits.flatten()
    else:
        predictions = np.argmax(logits, axis=-1)
    
    ### only for STS-binary
    if args.klue_task == "sts-binary":
        predictions = np.where(logits.flatten() > 3.0, 1, 0)
        labels = np.where(labels > 3.0, 1, 0)
        # print(predictions)
        # print(labels)
        # assert False

    # apply metric
    metric = load_metric(KLUE_TASKS[args.klue_task].metric.split("/")[0])
    if "/" in KLUE_TASKS[args.klue_task].metric:
        return metric.compute(predictions=predictions, 
                            references=labels, 
                            average=KLUE_TASKS[args.klue_task].metric.split("/")[1])
    else:
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
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_strategy="steps",
    logging_steps=500,
    report_to="tensorboard",
    logging_dir=f"{args.output_dir}/logs",
    load_best_model_at_end=True,
)


def load_model(args, inference=False):
    model = GPT2ForSequenceClassification.from_pretrained(args.pretrained_model, 
                                                          num_labels=3,
                                                          pad_token_id=0, 
                                                          cache_dir=args.cache_dir)
    if not inference:
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
        
        print(model)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"🥶 Frozen layer '{name}'")
            else:
                print(f"🚀 Trainable layer '{name}'")
    else:
        print("🔥 ==================== Inference: ==================== 🔥")
        assert args.pretrained_adapters_dir
        if args.finetune_strategies == "lang_adapters":
            adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/{args.adapter_lang_name}")
            model.set_active_adapters(adapter_name)
        elif args.finetune_strategies == "task_adapters":
            adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/{args.adapter_lang_name}")
            model.set_active_adapters(adapter_name)
            adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/xnli-task-adapter")
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