import numpy as np
import collections
import json

from transformers import set_seed

import torch

import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format="{level} {level.icon} | [{time}] - {message}")

from datasets import load_metric
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, AdapterTrainer, Seq2SeqAdapterTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str)
args = parser.parse_args()

language = args.lang
dataset = load_dataset("wikiann", language, cache_dir="/users/zyong2/data/zyong2/huggingface")

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

tok = "/users/zyong2/data/zyong2/bigscience/data/processed/020/tok_bloom-1b3_my_oscar_100000samples_24000vocab_extend"
model_name = "/users/zyong2/data/zyong2/bigscience/data/processed/020/bloom-1b3_my_bitfit_100000samples_24000vocab_extend"
# tok = model_name = 'bigscience/bloom-1b3'
# tok = model_name = 'sberbank-ai/mGPT'
# tok = model_name = 'bert-base-multilingual-cased'
# tok = model_name = 'xlm-roberta-large'

tokenizer = AutoTokenizer.from_pretrained(tok, cache_dir="/users/zyong2/data/zyong2/huggingface", add_prefix_space=True)
if not tokenizer.pad_token:
    if 'mGPT' in model_name:
        tokenizer.pad_token = '<pad>' #mGPT 
print(len(tokenizer))

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, max_length=128, padding="max_length", truncation=True)

    word_ids = tokenized_inputs.word_ids()  # Map tokens to their respective word.
    previous_word_idx = None
    label_ids = []
    for word_idx in word_ids:  # Set the special tokens to -100.
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:  # Only label the first token of a given word.
            label_ids.append(examples[f"ner_tags"][word_idx])
        else:
            label_ids.append(-100)
        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs

train_dataset = train_dataset.map(tokenize_function, batched=False)
val_dataset = val_dataset.map(tokenize_function, batched=False)
test_dataset = test_dataset.map(tokenize_function, batched=False)

metric = load_metric("seqeval")
idx2labelname = {i: label for i, label in enumerate(dataset["train"].features[f"ner_tags"].feature.names)}
def compute_metrics(eval_pred):
    logits, golds = eval_pred
    predictions = np.argmax(logits, axis=-1)

    converted_golds = list()
    converted_preds = list()

    for i in range(golds.shape[0]):
        gold, pred = list(), list()
        for j in range(golds.shape[1]):
            if golds[i][j] != -100:
                gold.append(idx2labelname[golds[i][j]])
                pred.append(idx2labelname[predictions[i][j]])
        converted_golds.append(gold)
        converted_preds.append(pred)

    metrics = metric.compute(predictions=converted_preds, references=converted_golds)
    def flatten(d):
        out = {}
        for key, val in d.items():
            if isinstance(val, dict):
                val = [val]
            if isinstance(val, list):
                for subdict in val:
                    deeper = flatten(subdict).items()
                    out.update({key + '_' + key2: val2 for key2, val2 in deeper})
            else:
                out[key] = val
        return out
    return flatten(metrics)

def print_model_trainable_layers(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"🥶 Frozen layer '{name}'")
        else:
            print(f"🚀 Trainable layer '{name}'")

scores = list()
for seed in range(2):
    set_seed(seed)
    
    if "madx" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained('bigscience/bloom-1b3', 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir="/users/zyong2/data/zyong2/huggingface",
                                                                    num_labels=7)
            model.add_adapter(f"{model_name}/oscar_pfeiffer+inv_{language}")
            model.set_active_adapters(f"{model_name}/oscar_pfeiffer+inv_{language}")

            model.add_adapter(f"xlsum-task-adapter")
            model.train_adapter(f"xlsum-task-adapter")
            print_model_trainable_layers(model)
            return model
    else:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(model_name, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir="/users/zyong2/data/zyong2/huggingface",
                                                                    num_labels=7)

            model.add_adapter(f"xlsum-task-adapter")
            model.train_adapter(f"xlsum-task-adapter")
            print_model_trainable_layers(model)
            return model

    # model.freeze_model(True)

    training_args = TrainingArguments(
        output_dir=f"/users/zyong2/data/zyong2/bigscience/data/processed/021-wikiann/pilot/{language}",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_steps=None,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_steps=500,
        report_to="tensorboard",
        logging_dir=f"/users/zyong2/data/zyong2/bigscience/data/processed/021-wikiann/pilot/{language}/logs",
        load_best_model_at_end=True, # will load the last saved **model** checkpoint, so will cause problem for adapters.
        metric_for_best_model='eval_overall_f1'
    )

    trainer = AdapterTrainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    with open(f"/users/zyong2/data/zyong2/bigscience/data/processed/021-wikiann/pilot/{language}/checkpoint-15/trainer_state.json") as rf:
        checkpoint = json.load(rf)['best_model_checkpoint']
        print(checkpoint)

    if "madx" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained('bigscience/bloom-1b3', 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir="/users/zyong2/data/zyong2/huggingface",
                                                                    num_labels=7)
            model.add_adapter(f"{model_name}/oscar_pfeiffer+inv_{language}")
            model.set_active_adapters(f"{model_name}/oscar_pfeiffer+inv_{language}")

            model.load_adapter(f"{checkpoint}/xlsum-task-adapter")
            model.set_active_adapters("xlsum-task-adapter")
            model.eval()
            return model
    else:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(model_name, 
                                                                pad_token_id=tokenizer.pad_token_id,
                                                                cache_dir="/users/zyong2/data/zyong2/huggingface",
                                                                num_labels=7)
            model.load_adapter(f"{checkpoint}/xlsum-task-adapter")
            model.set_active_adapters("xlsum-task-adapter")
            model.eval()
            return model
    
    eval_trainer = AdapterTrainer(
        model_init=model_init,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    res = eval_trainer.evaluate()
    print("Eval on Test set:", res)
    scores.append(res['eval_overall_f1'])

    torch.cuda.empty_cache()

print("Model:", model_name)
print(scores)
print(np.mean(scores) * 100)
print(np.std(scores) * 100)