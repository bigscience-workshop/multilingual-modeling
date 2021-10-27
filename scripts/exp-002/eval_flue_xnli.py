import logging
import argparse
import os

from datasets import load_dataset
from datasets import load_metric


import torch
import numpy as np
from transformers import TrainingArguments, Trainer
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification

# setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s ======   %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
logging.getLogger().addHandler(logging.StreamHandler())


parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
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
args = parser.parse_args()
if args.do_eval_after_train:
    args.do_predict = True

# load dataset
xnli_dataset = load_dataset("flue", "XNLI", cache_dir=args.cache_dir)
xnli_train_dataset = xnli_dataset['train']
xnli_val_dataset = xnli_dataset['validation']
xnli_test_dataset = xnli_dataset['test']


# load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer)

def tokenize_function(examples):
    return tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypo"]}', padding="max_length", truncation=True)

tokenizer.pad_token = tokenizer.eos_token  # tokenizer.encode(tokenizer.eos_token) = [0]
full_train_dataset = xnli_train_dataset.map(tokenize_function, batched=False)
full_val_dataset = xnli_val_dataset.map(tokenize_function, batched=False)
full_test_dataset = xnli_test_dataset.map(tokenize_function, batched=False)
small_train_dataset = full_train_dataset.shuffle(seed=42).select(range(100))
small_val_dataset = full_val_dataset.shuffle(seed=42).select(range(100))
small_test_dataset = full_test_dataset.shuffle(seed=42).select(range(100))

metric = load_metric("accuracy")
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
    logging_steps=1,
    report_to="tensorboard",
    logging_dir=f"{args.output_dir}/logs",
    load_best_model_at_end=True,
)

if args.do_train:
    model = GPT2ForSequenceClassification.from_pretrained(args.pretrained_model, 
                                                            num_labels=3, 
                                                            pad_token_id=0)
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=full_train_dataset, 
        eval_dataset=full_val_dataset, 
        compute_metrics=compute_metrics
    )

    trainer.train()

if args.do_eval_after_train:
    evaluation_dirs = list(sorted([
                checkpoint_dir
                for checkpoint_dir in os.listdir(args.output_dir)
                if checkpoint_dir.startswith('checkpoint-')
            ], key=lambda x: int(x[len('checkpoint-'):])))
    
    model = GPT2ForSequenceClassification.from_pretrained(evaluation_dirs[-1], 
                                                          num_labels=3, 
                                                          pad_token_id=0)

if args.do_predict:
    if args.do_eval_after_train:
        evaluation_dirs = list(sorted([
                    checkpoint_dir
                    for checkpoint_dir in os.listdir(args.output_dir)
                    if checkpoint_dir.startswith('checkpoint-')
                ], key=lambda x: int(x[len('checkpoint-'):])))
        args.pretrained_model = evaluation_dirs[-1]
        logging.info(f"Loading trained model from {evaluation_dirs[-1]}")
                                 
    model = GPT2ForSequenceClassification.from_pretrained(args.pretrained_model, 
                                                            num_labels=3, 
                                                            pad_token_id=0)
    trainer = Trainer(
        model=model, 
        args=training_args, 
        eval_dataset=full_test_dataset, 
        compute_metrics=compute_metrics
    )

    print("Evaluate:", trainer.evaluate())


