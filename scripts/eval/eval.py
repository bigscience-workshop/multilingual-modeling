import logging
import argparse
import os

from datasets import load_dataset
from datasets import load_metric
from collections import namedtuple

import nltk
import torch
import numpy as np
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, AdapterTrainer, Seq2SeqAdapterTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForSequenceClassification, AutoModelForCausalLM
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

# parser
parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--lang", type=str, default="german") #xlsum requires a language name, not language code

tasks = [XNLI, XLSUM]
parser.add_argument("--dataset", choices=tasks, required=True)

parser.add_argument("--cache_dir")
parser.add_argument("--num_train_epochs", type=int, default=30)
parser.add_argument("--max_steps", type=int, default=-1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--adapted_model") 
parser.add_argument("--original_model")  
parser.add_argument("--tokenizer")
parser.add_argument("--do_train", default=False, action="store_true")
parser.add_argument("--do_eval_after_train", default=False, action="store_true")
parser.add_argument("--do_predict", default=False, action="store_true")
parser.add_argument("--use_partial_data", default=False, action="store_true")
parser.add_argument("--use_partial_train_data", type=int, default=100)
parser.add_argument("--use_partial_val_data", type=int, default=-1)
parser.add_argument("--use_partial_test_data", type=int, default=-1)
parser.add_argument("--cross_lingual", default=False, action="store_true")
parser.add_argument("--revision", type=str, default="main")
parser.add_argument("--local_rank", type=int)

parser.add_argument("--madx_lang_adapter", default=None)
parser.add_argument("--no_task_adapter", default=False, action="store_true")
# parser.add_argument("--adapter_lang_name", required=True)
parser.add_argument("--deepspeed", required=False)

# mapping of tasks to model/trainer classes
model_class_mapping = {XNLI: AutoModelForSequenceClassification, XLSUM: AutoModelWithLMHead}
trainer_no_task_adpt_class_mapping = {XNLI: Trainer, XLSUM: Seq2SeqTrainer}
trainer_class_mapping = {XNLI: AdapterTrainer, XLSUM: Seq2SeqAdapterTrainer}
trainer_args_mapping = {XNLI: TrainingArguments, XLSUM: Seq2SeqTrainingArguments}


args = parser.parse_args()

#### Process args
if args.do_eval_after_train:
    args.do_predict = True

if not args.madx_lang_adapter and args.no_task_adapter:
    logger.warning("❗️ No 'madx_lang_adapter' loaded. This should be the baseline performance.")

# additional args to pass to the model init. task-dependent
optional_model_kwargs = {}
optional_trainer_args = {}
optional_eval_args = {}
if args.dataset == XNLI:
    optional_model_kwargs = {"num_labels": 3}
elif args.dataset == XLSUM:
    optional_trainer_args = {"generation_max_length": 512 + 64, 
                             "predict_with_generate":True,
                             "optim": "adafactor",
                             "lr_scheduler_type": "linear",
                             "warmup_steps": 0}


if args.local_rank:
    torch.cuda.set_device(args.local_rank)

if args.original_model is None:
    # here: because the wpe is not saved, adapted_model is the original bigsciece model
    args.original_model = args.adapted_model

print("Arguments: ========")
print(args)


# load appropriate dataset
logger.info("Loading dataset...")

# will need to rename splits if the dataset has different name for validation set
if args.cross_lingual:
    logger.info("0️⃣ Cross Lingual setting")
    # cross lingual: use english as train and validation set
    en_dataset = load_dataset(args.dataset, "english" if args.dataset == XLSUM else "en", cache_dir=args.cache_dir)
    dataset = load_dataset(args.dataset, args.lang, cache_dir=args.cache_dir)

    train_dataset = en_dataset["train"]
    val_dataset = en_dataset["validation"]
    test_dataset = dataset["test"]
else:
    logger.info("👀 Supervised training setting")
    dataset = load_dataset(args.dataset, args.lang, cache_dir=args.cache_dir)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

if args.use_partial_data:
    logger.warning("🚨 Loading partial data!")
    train_dataset = train_dataset.shuffle(seed=args.seed).select(range(args.use_partial_train_data))
    if args.use_partial_val_data != -1:
        val_dataset = val_dataset.shuffle(seed=args.seed).select(range(args.use_partial_val_data))
    if args.use_partial_test_data != -1:
        test_dataset = val_dataset.shuffle(seed=args.seed).select(range(args.use_partial_test_data))


# load tokenizer
logger.info("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, revision=args.revision)
tokenizer.pad_token = tokenizer.eos_token

# TODO: we probably need better code for this than multiple if-else statements
en_tokenizer = AutoTokenizer.from_pretrained(args.original_model, cache_dir=args.cache_dir, revision=args.revision)
en_tokenizer.pad_token = en_tokenizer.eos_token

if args.dataset == XNLI:
    def tokenize_function(examples):
        return tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

    def en_tokenize_function(examples):
        return en_tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

elif args.dataset == XLSUM:
    # for decoder only structure, input and target needs to have the same length
    # also, unlike enc-dec model, we cannot feed the model some text and expect the model to generate only summary 
    # we need to have input = [text] + [padding] and the output be [text] + [summary].
    def tokenize_function(example):
        text = tokenizer(f'{example["text"]}', max_length=511, truncation=True)
        # TODO: sep_token instead of bos_token
        input_text = tokenizer.decode(text['input_ids'], skip_special_tokens=True) + tokenizer.bos_token

        with tokenizer.as_target_tokenizer():
            summaries = tokenizer(f'{example["summary"]}', max_length=64, padding="max_length", truncation=True)
            summaries_text = tokenizer.decode(summaries['input_ids'], skip_special_tokens=True)
        
        inputs = tokenizer(f'{input_text + summaries_text}', max_length=512 + 64, padding="max_length", truncation=True)
        
        inputs["labels"] = inputs["input_ids"]

        return inputs


    def en_tokenize_function(example):
        inputs = en_tokenizer(f'{example["text"]}', max_length=512, padding="max_length", truncation=True)

        with en_tokenizer.as_target_tokenizer():
            summaries = en_tokenizer(f'{example["summary"]}', max_length=512, padding="max_length", truncation=True)
        
        inputs["labels"] = summaries["input_ids"]

        return inputs


# tokenizing the dataset
logger.info("Tokenizing the dataset...")
if args.do_train:
    if args.cross_lingual:
        train_dataset = train_dataset.map(en_tokenize_function, batched=False)
        val_dataset = val_dataset.map(en_tokenize_function, batched=False)
    else:
        train_dataset = train_dataset.map(tokenize_function, batched=False)
        val_dataset = val_dataset.map(tokenize_function, batched=False)

    logger.info("Print tokenized dataset example ...")
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

elif args.dataset == XLSUM:
    metric = load_metric('rouge')
    
    def compute_metrics(eval_preds):
        # TODO: note that this function calls trainer.model
        preds, labels = eval_preds

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
        
        result = metric.compute(predictions=preds, references=labels)
        # TODO: need to confirm these are the right rouge values to report. Can report more ROUGE metrics if needed.
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        return {k: round(v, 4) for k, v in result.items()}

    def compute_beam_search_metrics(model, dataset):
        input_ids = torch.Tensor(dataset['input_ids']).type(torch.IntTensor)
        model.cuda()
        print(input_ids.shape)
        print(model.device)

        beam_scorer = BeamSearchScorer(
            batch_size=2,
            num_beams=4,
            device=model.device,
        )

        # instantiate logits processors
        logits_processor = LogitsProcessorList(
            [
                ForcedEOSTokenLogitsProcessor(512+64, eos_token_id=model.config.eos_token_id),
            ]
        )

        preds = model.beam_search(input_ids[:2, :512].repeat_interleave(4, dim=0).cuda(), beam_scorer, logits_processor=logits_processor)
        preds = tokenizer.batch_decode(preds)
        print(preds)
        assert False
        labels = np.array(dataset['input_ids'])[:2, 512:]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
        result = metric.compute(predictions=preds, references=labels)
        print(result)
        # print(preds)
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        # labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
        
        # result = metric.compute(predictions=preds, references=labels)
        assert False

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
    deepspeed=args.deepspeed,
    **optional_trainer_args,
)

def load_model(args, inference=False):
    # load model
    pad_token_id = en_tokenizer.pad_token_id if (not inference and args.cross_lingual) else tokenizer.pad_token_id
    model = model_class_mapping[args.dataset].from_pretrained(args.original_model, 
                                                            pad_token_id=pad_token_id,
                                                            cache_dir=args.cache_dir,
                                                            revision=args.revision,
                                                            **optional_model_kwargs)

    if args.no_task_adapter:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if not inference and "word_embeddings" in name:
                param.requires_grad = True

            # print out for debugging
            if not param.requires_grad:
                print(f"Frozen layer '{name}'")
            else:
                print(f"Trainable layer '{name}'")
        return model


    # obtain the embeddings for training (supervised setting) and inference
    if "tr5b-1B3" in args.original_model: # previous 1.3B bigsience model
        token_embedding = torch.load(f'{args.adapted_model}/embedding_wte.pt')
        add_embedding = torch.load(f'{args.adapted_model}/embedding_wpe.pt')
    elif "bloom" in args.original_model:
        token_embedding = torch.load(f'{args.adapted_model}/word_embeddings.pt')
        add_embedding = torch.load(f'{args.adapted_model}/word_embeddings_layernorm.pt')

    # load embedding layers
    if not args.cross_lingual or inference:
        print(model.transformer.wte.weight)
        model.transformer.wte = wte
        print(model.transformer.wte.weight)
        assert False
        # need to load embedding/adapters from the model adapted to the new language
        causal_lm_model = AutoModelForCausalLM.from_pretrained(args.original_model)
        causal_lm_model.resize_token_embeddings(len(tokenizer))
        if not args.original_model == args.adapted_model:
            causal_lm_model.transformer.wte = wte
            causal_lm_model.transformer.wpe = wpe
        if args.madx_lang_adapter:
            adapter_name = causal_lm_model.load_adapter(args.madx_lang_adapter, config="pfeiffer+inv")                                    
            model.transformer = causal_lm_model.transformer
            model.set_active_adapters(adapter_name)

    if not inference:
        #if not args.cross_lingual: normally need to add adapter in any case
            # normally this is already done, why use adapter_lang_name here?
            #if args.madx_lang_adapter:
            #    adapter_name = model.load_adapter(args.madx_lang_adapter,
            #                                    config="pfeiffer+inv",
            #                                    load_as=args.adapter_lang_name)
        model.add_adapter(f"{args.dataset}-task-adapter")
        model.train_adapter(f"{args.dataset}-task-adapter")


        print("🔥 ==================== Training: ==================== 🔥")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                print(f"🥶 Frozen layer '{name}'")
            else:
                print(f"🚀 Trainable layer '{name}'")
        print(model)
    else:
        #if args.madx_lang_adapter:
        assert args.pretrained_adapters_dir 
            # normally this is done in any case
            #adapter_name = model.load_adapter(args.madx_lang_adapter)
            #model.set_active_adapters(adapter_name)
        adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/{args.dataset}-task-adapter")
        model.set_active_adapters(adapter_name)
        #else:
        #        # adapter_name = model.load_adapter("/users/zyong2/data/zyong2/bigscience/data/processed/013/xnli_de_de_100K_adpt_16_0shot/checkpoint-24544/xnli-task-adapter")
        #        # not sure what happens here
        #        # for TGT -> TGT supervised finetuning setting, change adapter_name
        #        adapter_name = model.load_adapter("/users/zyong2/data/zyong2/bigscience/data/processed/exp-013/task_xnli_de_ft_100000_ori/checkpoint-24544/xnli-task-adapter")
        #        model.set_active_adapters(adapter_name)
        print(model)
    return model


if args.do_train:
    logger.info("Starting training...")
    model = load_model(args)

    # only use seq2seq collator if doing seq2seq task
    if args.dataset == XLSUM:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
        )
    
    if args.no_task_adapter:
        logger.info(f"Using {trainer_no_task_adpt_class_mapping[args.dataset]} for training")
        trainer = trainer_no_task_adpt_class_mapping[args.dataset](
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            # args for xlsum only
            **{"data_collator": data_collator} if args.dataset == XLSUM else {},
        )
    else:
        logger.info(f"Using {trainer_class_mapping[args.dataset]} for training")
        trainer = trainer_class_mapping[args.dataset](
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
    logger.info(f"Found {len(evaluation_dirs)} checkpoints")

    # load the last checkpoint. 
    args.pretrained_adapters_dir = f"{args.output_dir}/{evaluation_dirs[-1]}"
    logger.info(f"[Evaluation] Loading trained model from {evaluation_dirs[-1]}")
        
    model = load_model(args, inference=True)
    training_args.report_to = list()

    if args.dataset == XLSUM:
        # use beam search to get the results following the XLSUM paper
        compute_beam_search_metrics(model, test_dataset)
        assert False

        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    if args.no_task_adapter:
        logger.info(f"Using {trainer_no_task_adpt_class_mapping[args.dataset]} for training")
        trainer = trainer_no_task_adpt_class_mapping[args.dataset](
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            # args for xlsum only
            **{"data_collator": data_collator} if args.dataset == XLSUM else {}
            )
    else:
        trainer = trainer_class_mapping[args.dataset](
            model=model,
            args=training_args,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
            # args for xlsum only
            **{"data_collator": data_collator} if args.dataset == XLSUM else {}

        )

    print("Evaluating on test set...", trainer.evaluate())
