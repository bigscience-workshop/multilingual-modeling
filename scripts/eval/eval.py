import logging
import argparse
import os
import json
from tqdm import tqdm

from datasets import load_dataset
from datasets import load_metric
from collections import namedtuple

import nltk
import torch
import numpy as np
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, AdapterTrainer, Seq2SeqAdapterTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForTokenClassification
from transformers import DataCollatorForSeq2Seq
from transformers import (
    get_linear_schedule_with_warmup,
    LogitsProcessorList,
    BeamSearchScorer,
    ForcedEOSTokenLogitsProcessor
)

# setup logging
import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format="{level} {level.icon} | [{time}] - {message}")


# AVAILABLE TASKS
XNLI = "xnli"
XLSUM = "csebuetnlp/xlsum"
WIKIANN = "wikiann"

# parser
parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--train_lang", type=str) 
parser.add_argument("--lang", type=str) #xlsum requires a language name, not language code

tasks = [XNLI, XLSUM, WIKIANN]
parser.add_argument("--dataset", choices=tasks, required=True)

parser.add_argument("--cache_dir")
parser.add_argument("--num_train_epochs", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--adapted_model_dir")
parser.add_argument("--original_model")  
parser.add_argument("--tokenizer")
parser.add_argument("--do_train", default=False, action="store_true")
parser.add_argument("--do_predict", default=False, action="store_true")
parser.add_argument("--use_partial_data", default=False, action="store_true")
parser.add_argument("--use_partial_train_data", type=int, default=100)
parser.add_argument("--use_partial_val_data", type=int, default=-1)
parser.add_argument("--use_partial_test_data", type=int, default=-1)
parser.add_argument("--cross_lingual", default=False, action="store_true")
parser.add_argument("--revision", type=str, default="main")
parser.add_argument("--local_rank", type=int)

parser.add_argument("--madx_lang_adapter", default=None)
parser.add_argument("--baseline", default=False, action="store_true")
parser.add_argument("--deepspeed", required=False)

task_layers = ["task-adapters", "last-layer", "full-model"]
parser.add_argument("--task_layers", choices=task_layers, required=True)


# mapping of tasks to model/trainer classes
model_class_mapping = {
    XNLI: AutoModelForSequenceClassification, 
    XLSUM: AutoModelWithLMHead,
    WIKIANN: AutoModelForTokenClassification
}
trainer_no_task_adpt_class_mapping = {XNLI: Trainer, XLSUM: Seq2SeqTrainer, WIKIANN: Trainer}
trainer_class_mapping = {XNLI: AdapterTrainer, XLSUM: Seq2SeqAdapterTrainer, WIKIANN: AdapterTrainer}
trainer_args_mapping = {XNLI: TrainingArguments, XLSUM: Seq2SeqTrainingArguments, WIKIANN: TrainingArguments}
task_eval_metric_best_model = {XNLI: 'eval_accuracy', XLSUM: 'eval_loss', WIKIANN: 'eval_overall_f1'}

args = parser.parse_args()

# XLSUM
XLSUM_INPUT_LEN = 512
XLSUM_OUTPUT_LEN = 64
XLSUM_NUM_BEAMS = 1
XLSUM_LEN_PENALTY = 0.6

#### Process args
if not args.cross_lingual and not args.train_lang:
    args.train_lang = args.lang
# ensure that only when cross_lingual, train_lang is not the same as lang
assert not ((args.train_lang != args.lang) ^ args.cross_lingual)

if args.baseline:
    logger.warning("❗️ No 'madx_lang_adapter' loaded. This should be the baseline performance.")
    assert not args.madx_lang_adapter

# additional args to pass to the model init. task-dependent
optional_model_kwargs = {}
optional_trainer_args = {}
optional_eval_args = {}
if args.dataset == XNLI:
    optional_model_kwargs = {"num_labels": 3}
elif args.dataset == WIKIANN:
    optional_model_kwargs = {"num_labels": 7}
elif args.dataset == XLSUM:
    optional_trainer_args = {"generation_max_length": XLSUM_INPUT_LEN + XLSUM_OUTPUT_LEN, 
                             "predict_with_generate":True,
                             "optim": "adafactor",
                             "lr_scheduler_type": "linear",
                             "warmup_ratio": 0.1}

if args.local_rank:
    torch.cuda.set_device(args.local_rank)

if args.original_model is None:
    # here: because the wpe is not saved, adapted_model_dir is the original bigsciece model
    args.original_model = args.adapted_model_dir

print("Arguments: ========")
print(args)

# load appropriate dataset
logger.info("Loading dataset...")

# will need to rename splits if the dataset has different name for validation set
if args.cross_lingual:
    logger.info(f"0️⃣ Cross Lingual setting")
    logger.info(f"train lang: {args.train_lang}; inference lang: {args.lang}")
    # cross lingual: use english as train and validation set
    en_dataset = load_dataset(args.dataset, args.train_lang, cache_dir=args.cache_dir)
    dataset = load_dataset(args.dataset, args.lang, cache_dir=args.cache_dir)

    train_dataset = en_dataset["train"]
    val_dataset = en_dataset["validation"]
    test_dataset = dataset["test"]
else:
    logger.info(f"👀 Supervised training setting")
    logger.info(f"language: {args.lang})")
    dataset = load_dataset(args.dataset, args.lang, cache_dir=args.cache_dir)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

if args.use_partial_data:
    train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.use_partial_train_data))
    if args.use_partial_val_data != -1:
        val_dataset = val_dataset.shuffle(seed=args.seed).select(range(args.use_partial_val_data))
    if args.use_partial_test_data != -1:
        test_dataset = test_dataset.shuffle(seed=args.seed).select(range(args.use_partial_test_data))
    logger.warning("🚨 Loading partial data!")

if args.do_train:
    logger.info(f"train = {len(train_dataset)} samples")
else:
    logger.info(f"args.do_train = False")
logger.info(f"val = {len(val_dataset)} samples")
logger.info(f"test = {len(test_dataset)} samples")

# load tokenizer
logger.info(f"Loading tokenizer from {args.tokenizer}...")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, revision=args.revision, add_prefix_space=args.dataset in [WIKIANN])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
if tokenizer.sep_token is None:
    tokenizer.sep_token = tokenizer.bos_token

# TODO: we probably need better code for this than multiple if-else statements
en_tokenizer = AutoTokenizer.from_pretrained(args.original_model, cache_dir=args.cache_dir, revision=args.revision, add_prefix_space=args.dataset in [WIKIANN])
if en_tokenizer.pad_token is None:
    en_tokenizer.pad_token = en_tokenizer.eos_token
if en_tokenizer.sep_token is None:
    en_tokenizer.sep_token = en_tokenizer.bos_token
    # en_tokenizer.add_special_tokens({'sep_token':'<|sep|>'})

if args.dataset == XNLI:
    if tokenizer.eos_token is None:
        tokenizer.eos_token = tokenizer.sep_token
    if en_tokenizer.eos_token is None:
        en_tokenizer.eos_token = en_tokenizer.sep_token

    def tokenize_function(examples):
        return tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

    def en_tokenize_function(examples):
        return en_tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

elif args.dataset == XLSUM:
    # for decoder only structure, input and target needs to have the same length
    # also, unlike enc-dec model, we cannot feed the model some text and expect the model to generate only summary 
    # we need to have input = [text] + [padding] and the output be [text] + [summary].
    def tokenize_function(example):
        text = tokenizer(f'{example["text"]}', max_length=XLSUM_INPUT_LEN - 1, padding="max_length", truncation=True)
        input_text = tokenizer.decode(text['input_ids'], skip_special_tokens=False) + tokenizer.sep_token

        with tokenizer.as_target_tokenizer():
            summaries = tokenizer(f'{example["summary"]}', max_length=XLSUM_OUTPUT_LEN, padding="max_length", truncation=True)
            summaries_text = tokenizer.decode(summaries['input_ids'], skip_special_tokens=False)
        
        inputs = tokenizer(f'{input_text + summaries_text}')
        inputs["labels"] = inputs["input_ids"]

        return inputs

    def en_tokenize_function(example):
        ...
        # inputs = en_tokenizer(f'{example["text"]}', max_length=512, padding="max_length", truncation=True)

        # with en_tokenizer.as_target_tokenizer():
        #     summaries = en_tokenizer(f'{example["summary"]}', max_length=512, padding="max_length", truncation=True)
        
        # inputs["labels"] = summaries["input_ids"]

        # return inputs

elif args.dataset == WIKIANN:
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

    def en_tokenize_function(examples):
        return en_tokenizer(examples['tokens'], is_split_into_words=True, max_length=128, padding="max_length", truncation=True)


# tokenizing the dataset
logger.info("Tokenizing the dataset...")
if args.do_train:
    if args.cross_lingual:
        train_dataset = train_dataset.map(en_tokenize_function, batched=False)
        val_dataset = val_dataset.map(en_tokenize_function, batched=False)
    else:
        train_dataset = train_dataset.map(tokenize_function, batched=False)
        val_dataset = val_dataset.map(tokenize_function, batched=False)

    logger.info("Print one tokenized dataset example ...")
    logger.info(train_dataset[0])

test_dataset = test_dataset.map(tokenize_function, batched=False)

# TODO: same as above, we probably need a better way than if-else statements.
# load metric
logger.info("Loading metric...")

if args.dataset == XNLI:
    metric = load_metric("xnli")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

elif args.dataset == WIKIANN:
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

        return metric.compute(predictions=converted_preds, references=converted_golds)

elif args.dataset == XLSUM:
    metric = load_metric('rouge')
    
    def compute_metrics(eval_preds):
        return {}

    def compute_xlsum_beam_search_metrics(model, dataset):
        # get input sentences
        input_ids = torch.Tensor(dataset['input_ids']).type(torch.IntTensor)[:, :XLSUM_INPUT_LEN]
        bsz = args.per_device_eval_batch_size

        # get generated summaries
        preds = list()
        for i in tqdm(range(0, input_ids.shape[0], bsz), desc="Summarization task: generation"):
            outputs = model.generate(input_ids[i:i+bsz], max_length=XLSUM_INPUT_LEN+XLSUM_OUTPUT_LEN, length_penalty=XLSUM_LEN_PENALTY, num_beams=XLSUM_NUM_BEAMS)
            preds += tokenizer.batch_decode(outputs[:, XLSUM_INPUT_LEN:], skip_special_tokens=True)

        # get gold summaries
        labels = np.array(dataset['input_ids'])[:, XLSUM_INPUT_LEN:]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        print(preds)
        print(labels)

        # compute ROUGE metrics
        preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
        result = metric.compute(predictions=preds, references=labels)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        return {k: round(v, 4) for k, v in result.items()}

else:
    raise ValueError("Unknown dataset provided")


training_args = trainer_args_mapping[args.dataset](
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_steps=500 if not args.use_partial_data else None,
    num_train_epochs=args.num_train_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    logging_steps=500,
    report_to="tensorboard",
    logging_dir=f"{args.output_dir}/logs",
    load_best_model_at_end=True,
    metric_for_best_model=task_eval_metric_best_model[args.dataset],
    deepspeed=args.deepspeed,
    **optional_trainer_args,
)

def print_model_trainable_layers(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"🥶 Frozen layer '{name}'")
        else:
            print(f"🚀 Trainable layer '{name}'")

def load_model(args, inference=False):
    def make_last_layer_trainable(args, model, inference=False):
        if model is None:
            if not inference:
                model_path = args.original_model
            else:
                model_path = args.pretrained_adapters_dir
            print(f"Loaded model from {model_path}")
            model = model_class_mapping[args.dataset].from_pretrained(model_path, 
                                                                      pad_token_id=pad_token_id,
                                                                      cache_dir=args.cache_dir,
                                                                      revision=args.revision,
                                                                      **optional_model_kwargs)
        model.freeze_model(freeze=True)
        return model
    
    def make_base_model_trainable(args, model, inference=False):
        if model is None:
            if not inference:
                model_path = args.original_model
            else:
                model_path = args.pretrained_adapters_dir
            print(f"Loaded model from {model_path}")
            model = model_class_mapping[args.dataset].from_pretrained(model_path, 
                                                                      pad_token_id=pad_token_id,
                                                                      cache_dir=args.cache_dir,
                                                                      revision=args.revision,
                                                                      **optional_model_kwargs)
        model.freeze_model(freeze=False)
        return model

    def load_task_specific_adapters(args, model, inference=False):
        if model is None:
            model = model_class_mapping[args.dataset].from_pretrained(args.original_model, 
                                                                      pad_token_id=pad_token_id,
                                                                      cache_dir=args.cache_dir,
                                                                      revision=args.revision,
                                                                      **optional_model_kwargs)
        
        if not inference:
            model.add_adapter(f"{args.dataset.split('/')[-1]}-task-adapter")
            model.train_adapter(f"{args.dataset.split('/')[-1]}-task-adapter")
            return model
        
        else:
            print(f"[Evaluation] Load task adapters from {args.pretrained_adapters_dir}/{args.dataset.split('/')[-1]}-task-adapter")
            adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/{args.dataset.split('/')[-1]}-task-adapter")
            model.set_active_adapters(adapter_name)
            return model

    def load_embedding_layers(args, tokenizer, model):
        if "tr5b-1B3" in args.original_model: # previous 1.3B bigsience model
            token_embedding = torch.load(f'{args.adapted_model_dir}/embedding_wte.pt')
            add_embedding = torch.load(f'{args.adapted_model_dir}/embedding_wpe.pt')
            model.transformer.wte = token_embedding
            model.transformer.wpe = add_embedding
        
        elif "bloom" in args.original_model:
            token_embedding = torch.load(f'{args.adapted_model_dir}/word_embeddings.pt')
            add_embedding = torch.load(f'{args.adapted_model_dir}/word_embeddings_layernorm.pt')
            model.transformer.word_embeddings = token_embedding
            model.transformer.word_embeddings_layernorm = add_embedding

        logger.info(f"Replaced embeddings with {token_embedding} and {add_embedding}...")
        return model
    
    def load_language_adapters(args, model):
        adapter_name = model.load_adapter(args.madx_lang_adapter, config="pfeiffer+inv")
        model.set_active_adapters(adapter_name)
        logger.info(f"Added Adapter {args.madx_lang_adapter}...")
        return model

    pad_token_id = en_tokenizer.pad_token_id if (not inference and args.cross_lingual) else tokenizer.pad_token_id

    # baseline: only need to add task-specific adapters 
    # (keeps separated for now for easier debugging)
    if args.baseline:
        model = None
        if args.task_layers == "task-adapters":
            model = load_task_specific_adapters(args, model, inference)
        elif args.task_layers == "last-layer":
            model = make_last_layer_trainable(args, model, inference)
        elif args.task_layers == "full-model":
            model = make_base_model_trainable(args, model, inference)
        return model

    # load unadapted model
    model = model_class_mapping[args.dataset].from_pretrained(args.original_model, 
                                                              pad_token_id=pad_token_id,
                                                              cache_dir=args.cache_dir,
                                                              revision=args.revision,
                                                              **optional_model_kwargs)
                                                              
    # load adapted model
    if not args.cross_lingual or inference:
        model = load_embedding_layers(args, tokenizer, model)
        if args.madx_lang_adapter:
            model = load_language_adapters(args, model)
    
    if args.task_layers == "task-adapters":
        model = load_task_specific_adapters(args, model, inference)
    elif args.task_layers == "last-layer":
        model = make_last_layer_trainable(args, model, inference)
    return model


if args.do_train:
    logger.info("Starting training...")
    model = load_model(args)
    print("🔥 ==================== Training: ==================== 🔥")
    print_model_trainable_layers(model)

    # only use seq2seq collator if doing seq2seq task
    if args.dataset == XLSUM:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
        )
    
    if model.active_adapters is None:
        logger.info("No active adapters")
        trainer_class = trainer_no_task_adpt_class_mapping[args.dataset]
    else:
        trainer_class = trainer_class_mapping[args.dataset]
    logger.info(f"Using {trainer_class_mapping[args.dataset]} for training")
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        # args for xlsum only
        **{"data_collator": data_collator} if args.dataset == XLSUM else {},
    )

    trainer.train()


if args.do_predict:
    evaluation_dirs = list(sorted([
        checkpoint_dir 
        for checkpoint_dir in os.listdir(args.output_dir) 
        if checkpoint_dir.startswith("checkpoint-")],
        key=lambda x: int(x[len('checkpoint-'):])))
    assert len(evaluation_dirs) > 0
    print(f"Found {len(evaluation_dirs)} checkpoints")

    # load the best checkpoint. 
    with open(f"{args.output_dir}/{evaluation_dirs[-1]}/trainer_state.json") as rf:
        args.pretrained_adapters_dir = json.load(rf)['best_model_checkpoint']

    print(f"[Evaluation] Loading trained model (best checkpoint) from {args.pretrained_adapters_dir}")
        
    model = load_model(args, inference=True)
    model.eval()
    training_args.report_to = list()

    if args.dataset == XLSUM:
        # use beam search to get the results following the XLSUM paper
        print(f"Evaluating on test set ({XLSUM})...")
        result = compute_xlsum_beam_search_metrics(model, test_dataset)
        print(result)
    
    else:
        if model.active_adapters is None:
            logger.info("No active adapters")
            trainer_class = trainer_no_task_adpt_class_mapping[args.dataset]
        else:
            trainer_class = trainer_class_mapping[args.dataset]

        eval_trainer = trainer_class(
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            # args for xlsum only
            **{"data_collator": data_collator} if args.dataset == XLSUM else {}

        )

        print("Evaluating on test set...")
        print(eval_trainer.evaluate())

