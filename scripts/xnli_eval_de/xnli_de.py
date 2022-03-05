import logging
import argparse
import os

from datasets import load_dataset
from datasets import load_metric
from collections import namedtuple

import torch
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import AutoTokenizer, GPT2Config, GPT2Tokenizer, GPT2ForSequenceClassification

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
args = parser.parse_args()
if args.do_eval_after_train:
    args.do_predict = True

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
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    report_to="tensorboard",
    logging_dir=f"{args.output_dir}/logs",
    load_best_model_at_end=True,
)

def load_model(pretrained_model, cache_dir, pad_token_id=0):
    return GPT2ForSequenceClassification.from_pretrained(pretrained_model, 
                                                         num_labels=3,
                                                         pad_token_id=pad_token_id, 
                                                         cache_dir=cache_dir)


if args.do_train:
    logger.info("Start Training")
    model = load_model(args.pretrained_model, 
                       args.cache_dir, 
                       en_tokenizer.pad_token_id if args.zero_shot else tokenizer.pad_token_id)

    if args.zero_shot:
        # model is the finetuned model
        original_config = GPT2Config.from_pretrained(args.original_model)
        original_model = load_model(args.original_model, args.cache_dir)
        no_en_wte = model._modules['transformer']._modules['wte']
        no_en_wpe = model._modules['transformer']._modules['wpe']

        # replace the embedding layer with original (contains-en) embedding.
        logger.info("👉 Replace with en-langauge embedding")
        model.resize_token_embeddings(original_config.vocab_size)
        
        model._modules['transformer']._modules['wte'] = original_model._modules['transformer']._modules['wte']
        model._modules['transformer']._modules['wpe'] = original_model._modules['transformer']._modules['wpe']
        logger.info(f"👉 Embedding (wte) changes from {no_en_wte} to {model._modules['transformer']._modules['wte']}")
        logger.info(f"👉 Embedding (wte) changes from {no_en_wpe} to {model._modules['transformer']._modules['wpe']}")

    trainer = Trainer(
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
        args.pretrained_model = f"{args.output_dir}/{evaluation_dirs[-1]}"
        logger.info(f"[Evaluation] Loading trained model from {evaluation_dirs[-1]}")

    # FIXME: hack for now because the tokenizer loaded from bigscience doesn't have the same
    # vocab size as indicated in the config.json
    # not optimal fix for now because cooriginal_confignfig can be directly passed to from_pretrained
    if args.zero_shot:
        original_config.save_pretrained(args.pretrained_model)
    
    model = load_model(args.pretrained_model, args.cache_dir, tokenizer.pad_token_id)
    if args.zero_shot:
        # replace with target-language embedding.
        logger.info("👉 Replace with target-language embedding")
        logger.info(f"👉 len(tokenizer) = {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        model._modules['transformer']._modules['wte'] = no_en_wte
        model._modules['transformer']._modules['wpe'] = no_en_wpe

    training_args.report_to = list()
    
    trainer = Trainer(
        model=model, 
        args=training_args, 
        eval_dataset=small_test_dataset if args.use_partial_data else full_test_dataset, 
        compute_metrics=compute_metrics
    )

    print("Evaluate on Test:", trainer.evaluate())