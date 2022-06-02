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
from transformers.adapters.configuration import AdapterConfig
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
parser.add_argument("--per_device_train_batch_size", type=int, default=8)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--pretrained_model") 
parser.add_argument("--original_model")  
parser.add_argument("--tokenizer")
parser.add_argument("--do_train", default=False, action="store_true")
parser.add_argument("--do_eval_after_train", default=False, action="store_true")
parser.add_argument("--do_predict", default=False, action="store_true")
parser.add_argument("--use_partial_data", default=None, type=int)
parser.add_argument("--cross_lingual", default=False, action="store_true")

finetune_strategies = ["whole", "lang_adapters", "task_adapters", "classification_head", "task_adapter_last_layer"]
parser.add_argument("--madx_lang_adapter")
#parser.add_argument("--adapter_lang_name", required=True) -- why is this required??
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
if args.cross_lingual:
    print("0️⃣ 0-Shot")
    # 0-shot: use english as train and validation
    if args.do_train:
        xnli_en_dataset = load_dataset("xnli", "en", cache_dir=args.cache_dir)
    xnli_dataset = load_dataset("xnli", args.lang, cache_dir=args.cache_dir)
#    assert args.lang != "en"
    if args.do_train:
        train_dataset = xnli_en_dataset['train']
        val_dataset = xnli_en_dataset['validation']
    test_dataset = xnli_dataset['test']
else:
    print("👀 Supervised Training")
    xnli_dataset = load_dataset("xnli", args.lang, cache_dir=args.cache_dir)

    train_dataset = xnli_dataset['train']
    val_dataset = xnli_dataset['validation']
    test_dataset = xnli_dataset['test']


en_tokenizer = AutoTokenizer.from_pretrained(args.original_model, cache_dir=args.cache_dir) # has to use AutoTokenizer instead of GPT2Tokenizer
en_tokenizer.pad_token = en_tokenizer.eos_token

def en_tokenize_function(examples):
    return en_tokenizer(f'{examples["premise"]} {en_tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)
if args.tokenizer:
    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token  # tokenizer.encode(tokenizer.eos_token) = [0]

    def tokenize_function(examples):
        return tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)
else:
    def tokenize_function(examples):
        return en_tokenize_function(examples)
    tokenizer = en_tokenizer
    



logger.info("Tokenizing the dataset...")
if args.do_train:
    if args.cross_lingual:
        full_train_dataset = train_dataset.map(en_tokenize_function, batched=False)
        full_val_dataset = val_dataset.map(en_tokenize_function, batched=False)
    else:
        full_train_dataset = train_dataset.map(tokenize_function, batched=False)
        full_val_dataset = val_dataset.map(tokenize_function, batched=False)

if args.use_partial_data:
    small_train_dataset = full_train_dataset.shuffle(seed=42).select(range(args.use_partial_data))
    small_val_dataset = full_val_dataset.shuffle(seed=42).select(range(min(1000, args.use_partial_data)))
    logger.info(full_train_dataset[0])
    logger.info(full_train_dataset[100])

full_test_dataset = test_dataset.map(tokenize_function, batched=False)
small_test_dataset = full_test_dataset.shuffle(seed=42).select(range(100))

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
    fp16=True,
    fp16_full_eval=True,
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
        
    model = GPT2ForSequenceClassification.from_pretrained(args.original_model, 
                                                          num_labels=3,
                                                          pad_token_id=en_tokenizer.pad_token_id,
                                                          cache_dir=args.cache_dir)

    if not args.cross_lingual or inference:
        # need to load embedding/adapters from the model adapted to the new language
        causal_lm_model = AutoModelForCausalLM.from_pretrained(args.original_model)
        causal_lm_model.resize_token_embeddings(len(tokenizer))
        if args.madx_lang_adapter:
            lang_adapter_name = causal_lm_model.load_adapter(args.madx_lang_adapter, config="pfeiffer+inv")                                    
            model.transformer = causal_lm_model.transformer
            model.set_active_adapters(adapter_name)
        if not args.original_model == args.pretrained_model:
            wte = torch.load(f'{args.pretrained_model}/embedding.pt')
            wpe = torch.load(f'{args.pretrained_model}/positional_embedding.pt')        
            model.transformer.wte.weight.data = wte
            model.transformer.wpe.weight.data = wpe


    if not inference:
        if args.finetune_strategies == "task_adapters":
            model.add_adapter("xnli-task-adapter")
            model.train_adapter("xnli-task-adapter")

        if args.finetune_strategies == "task_adapter_last_layer":
            adapter_config = AdapterConfig.load(
                "pfeiffer+inv",
                reduction_factor=16,
                leave_out = [i for i in range(0,23)]
            )
            model.add_adapter("xnli-task-adapter-ll", config=adapter_config)
            model.train_adapter("xnli-task-adapter-ll")
            

        print("🔥 ==================== Training: ==================== 🔥")
        for name, param in model.named_parameters():
            if args.finetune_strategies == "whole":
                param.requires_grad = True
            elif args.finetune_strategies == "classification_head":
                if name == "score.weight":
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            if not param.requires_grad:
                print(f"🥶 Frozenlayer '{name}'")
            else:
                print(f"🚀 Trainable layer '{name}'")
        print(model)

    else:
        if args.finetune_strategies == "task_adapters":
            assert args.pretrained_adapters_dir 
            adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/xnli-task-adapter")
            model.set_active_adapters(adapter_name)            
        elif args.finetune_strategies == "task_adapter_last_layer":
            assert args.pretrained_adapters_dir 
            adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/xnli-task-adapter-ll")
            model.set_active_adapters(adapter_name)

        print(model)
        #Need to make sure that we use correct embeddings and adapter -- TBC?
        model.transformer.wte.weight.data = wte
        model.transformer.wpe.weight.data = wpe


    return model

if args.do_train:
    logger.info("Start Training")
    model = load_model(args)
    if args.finetune_strategies in ["task_adapters", "task_adapter_last_layer"] :
        trainer = AdapterTrainer(
            model=model, 
            args=training_args, 
            train_dataset=small_train_dataset if args.use_partial_data else full_train_dataset, 
            eval_dataset=small_val_dataset if args.use_partial_data else full_val_dataset, 
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=small_train_dataset if args.use_partial_data else full_train_dataset, 
            eval_dataset=small_val_dataset if args.use_partial_data else full_val_dataset, 
            compute_metrics=compute_metrics
        )

    trainer.train()
#    trainer.save_model()

if args.do_predict and not args.do_train:
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
    if args.finetune_strategies in ["task_adapters", "task_adapter_last_layer"]:
        trainer = AdapterTrainer(
            model=model, 
            args=training_args, 
            eval_dataset=full_test_dataset, 
            compute_metrics=compute_metrics
        )
    else:
        trainer = Trainer(
            model=model, 
            args=training_args, 
            train_dataset=small_train_dataset if args.use_partial_data else full_train_dataset, 
            eval_dataset=full_test_dataset, 
            compute_metrics=compute_metrics
        )

    print("Evaluate on Test:", trainer.evaluate())

elif args.do_predict and args.do_train:
    print("Evaluate on Test:", trainer.evaluate())
