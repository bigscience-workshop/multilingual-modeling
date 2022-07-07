import argparse
import os
import sys
from loguru import logger

from datasets import load_dataset
from datasets import load_metric

import torch
import numpy as np
import nltk
from transformers import TrainingArguments, AdapterTrainer, Seq2SeqAdapterTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2ForSequenceClassification, AutoModelForCausalLM
from transformers import DataCollatorForSeq2Seq


logger.remove()
logger.add(sys.stderr, format="{level} {level.icon} | [{time}] - {message}")

# parser
parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--lang", type=str, default="german") #xlsum requires a language name, not language code

tasks = ["xnli", "xlsum"]
parser.add_argument("--dataset", choices=tasks, required=True)

parser.add_argument("--cache_dir")
parser.add_argument("--num_train_epochs", type=int, default=30)
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--per_device_train_batch_size", type=int, default=4)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
parser.add_argument("--pretrained_model") 
parser.add_argument("--original_model")  
parser.add_argument("--tokenizer")
parser.add_argument("--do_train", default=False, action="store_true")
parser.add_argument("--do_eval_after_train", default=False, action="store_true")
parser.add_argument("--do_predict", default=False, action="store_true")
parser.add_argument("--use_partial_data", default=False, action="store_true")
parser.add_argument("--zero_shot", default=False, action="store_true")
parser.add_argument("--revision", type=str, default="main")
parser.add_argument("--local_rank", type=int)

finetune_strategies = ["whole", "lang_adapters", "task_adapters"]
parser.add_argument("--madx_lang_adapter")
parser.add_argument("--adapter_lang_name", required=True)
parser.add_argument("--finetune_strategies", choices=finetune_strategies, required=True)

parser.add_argument("--deepspeed", required=False)

# mapping of tasks to model/trainer classes
model_class_mapping = {"xnli": GPT2ForSequenceClassification, "xlsum": GPT2LMHeadModel}
trainer_class_mapping = {"xnli": AdapterTrainer, "xlsum": Seq2SeqAdapterTrainer}
trainer_args_mapping = {"xnli": TrainingArguments, "xlsum": Seq2SeqTrainingArguments}


args = parser.parse_args()
if args.do_eval_after_train:
    args.do_predict = True

# additional args to pass to the model init. task-dependent
optional_model_kwargs = {}
optional_trainer_args = {}
if args.dataset == "xnli":
    optional_model_kwargs = {"num_labels": 3}
elif args.dataset == "xlsum":
    optional_trainer_args = {"generation_max_length": 128, "predict_with_generate":True}


if args.local_rank:
    torch.cuda.set_device(args.local_rank)

if args.original_model is None:
    # here: because the wpe is not saved, pretrained_model is the original bigscience model
    args.original_model = args.pretrained_model

print("Arguments: ========")
print(args)

# load appropriate dataset
logger.info("Loading dataset...")

# will need to rename splits if the dataset has different name for validation set
if args.zero_shot:
    print("0️⃣ Cross Lingual")
    # cross lingual: use english as train and validation set
    en_dataset = load_dataset(args.dataset, "english" if args.dataset == "xlsum" else "en", cache_dir=args.cache_dir)
    dataset = load_dataset(args.dataset, args.lang, cache_dir=args.cache_dir)

    train_dataset = en_dataset["train"]
    val_dataset = en_dataset["validation"]
    test_dataset = dataset["test"]
else:
    print("👀 Supervised training")
    dataset = load_dataset(args.dataset, args.lang, cache_dir=args.cache_dir)

    train_dataset = dataset["train"]
    val_dataset = dataset["validation"]
    test_dataset = dataset["test"]

logger.info("Loading tokenizer...")
# load tokenizer

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, revision=args.revision)
tokenizer.pad_token = tokenizer.eos_token

if args.dataset == "xnli":
    def tokenize_function(examples):
        return tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

elif args.dataset == "xlsum":
    def tokenize_function(example):
        inputs = tokenizer(f'summarize this article: {example["text"]}', max_length=96, padding="max_length", truncation=True)

        with tokenizer.as_target_tokenizer():
            summaries = tokenizer(f'{example["summary"]}', max_length=96, padding="max_length", truncation=True)
        
        inputs["labels"] = summaries["input_ids"]

        return inputs

if args.zero_shot:
    en_tokenizer = AutoTokenizer.from_pretrained(args.original_model, cache_dir=args.cache_dir, revision=args.revision)
    en_tokenizer.pad_token = en_tokenizer.eos_token
    
    if args.dataset == "xnli":
        def en_tokenize_function(examples):
            return en_tokenizer(f'{examples["premise"]} {tokenizer.eos_token} {examples["hypothesis"]}', max_length=128, padding="max_length", truncation=True)

    elif args.dataset == "xlsum":
        def en_tokenize_function(example):
            inputs = en_tokenizer(f'summarize this article: {example["text"]}', max_length=96, padding="max_length", truncation=True)

            with en_tokenizer.as_target_tokenizer():
                summaries = en_tokenizer(f'{example["summary"]}', max_length=96, padding="max_length", truncation=True)
            
            inputs["labels"] = summaries["input_ids"]

            return inputs



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


# load metric
logger.info("Loading metric...")

if args.dataset == "xnli":
    metric = load_metric("xnli")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

elif args.dataset == "xlsum":
    metric = load_metric("rouge", cache_dir=args.cache_dir)

    def compute_metrics(eval_preds):
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

else:
    raise ValueError("Unknown dataset provided")


training_args = trainer_args_mapping[args.dataset](
    output_dir=args.output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_steps=500 if not args.use_partial_data else None,
    num_train_epochs=args.num_train_epochs,
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

# TODO: double-check the adapter loading logic here
def load_model(args, inference=False):

    # Hack for loading wte module not needed here, since using a causal language model class
    if args.zero_shot and not inference:
        # only pass in num_labels if using a seq. classification model
        model = model_class_mapping[args.dataset].from_pretrained(args.pretrained_model, 
                                                            pad_token_id=en_tokenizer.pad_token_id,
                                                            cache_dir=args.cache_dir,
                                                            revision=args.revision,
                                                            **optional_model_kwargs)
    else:
        model = model_class_mapping[args.dataset].from_pretrained(args.pretrained_model,
                                                            pad_token_id=tokenizer.pad_token_id,
                                                            cache_dir=args.cache_dir,
                                                            revision=args.revision,
                                                            **optional_model_kwargs)
    if not args.zero_shot or (args.zero_shot and inference):
        # if not zero shot, that means that we need to replace the embedding layers during training
        # we also need to replace embedding layers during inference
        causal_lm_model = AutoModelForCausalLM.from_pretrained(args.original_model, revision=args.revision)

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
            model.add_adapter(f"{args.dataset}-task-adapter")
            model.train_adapter(f"{args.dataset}-task-adapter")
        else:
            raise ValueError("invalid configuration")
        
        print("🔥 ==================== Training: ==================== 🔥")
        # for name, param in model.named_parameters():
        #     if not param.requires_grad:
        #         print(f"🥶 Frozen layer '{name}'")
        #     else:
        #         print(f"🚀 Trainable layer '{name}'")
        # print(model)
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
                adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/{args.dataset}-task-adapter")
                model.set_active_adapters(adapter_name)
            else:
                adapter_name = model.load_adapter(f"{args.pretrained_adapters_dir}/{args.dataset}-task-adapter") #TODO: change the argument to this
                model.set_active_adapters(adapter_name)
        # print(model)


    return model


if args.do_train:
    logger.info("Starting training...")
    model = load_model(args)

    
    # only use seq2seq collator if doing seq2seq task
    if args.dataset == "xlsum":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
        )
    

    trainer = trainer_class_mapping[args.dataset](
        model=model,
        args=training_args,
        train_dataset=small_train_dataset if args.use_partial_data else full_train_dataset,
        eval_dataset=small_val_dataset if args.use_partial_data else full_val_dataset,
        compute_metrics=compute_metrics,
        # args for xlsum only
        **{"data_collator": data_collator} if args.dataset == "xlsum" else {},
    )

    trainer.train()



if args.do_predict:
    if args.do_eval_after_train:
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

    if args.dataset == "xlsum":
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    
    trainer = trainer_class_mapping[args.dataset](
        model=model,
        args=training_args,
        eval_dataset=small_test_dataset if args.use_partial_data else full_test_dataset,
        compute_metrics=compute_metrics,
        # args for xlsum only
        **{"data_collator": data_collator} if args.dataset == "xlsum" else {}

    )

    print("Evaluating on test set...", trainer.evaluate())
