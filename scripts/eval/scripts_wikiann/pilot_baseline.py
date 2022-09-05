import numpy as np
import collections
import json
import pathlib
import gc

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
from transformers.adapters.composition import Stack
from transformers import AdapterConfig, LoRAConfig, PrefixTuningConfig, ConfigUnion, IA3Config


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str)
parser.add_argument("--cache_dir", type=str, default="/users/zyong2/data/zyong2/huggingface")
parser.add_argument("--output_dir", type=str)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--base_model", type=str, default="bigscience/bloom-1b3")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--reproducible", action="store_true")
parser.add_argument("--seed_runs", type=int, default=10)
args = parser.parse_args()

language = args.lang
dataset = load_dataset("wikiann", language, cache_dir=args.cache_dir)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]
test_dataset = dataset["test"]

tok = args.tokenizer
model_name = args.model_name
base_model = args.base_model



tokenizer = AutoTokenizer.from_pretrained(tok, cache_dir=args.cache_dir, add_prefix_space=True)
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

metric = load_metric("seqeval", experiment_id=model_name.replace("/", "_"))  # add experiment_id to prevent conflicting access
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
        
        print(param)
    
    print(model)


task_config_dict = {
    "adapter_residual_before_ln": False,
    "cross_adapter": False,
    "factorized_phm_W": True,
    "factorized_phm_rule": False,
    "hypercomplex_nonlinearity": "glorot-uniform",
    "init_weights": "bert",
    "is_parallel": False,
    "learn_phm": True,
    "leave_out": [],
    "ln_after": False,
    "ln_before": False,
    "mh_adapter": False,
    "non_linearity": "relu",
    "original_ln_after": True,
    "original_ln_before": True,
    "output_adapter": True,
    "phm_bias": True,
    "phm_c_init": "normal",
    "phm_dim": 4,
    "phm_init_range": 0.0001,
    "phm_layer": False,
    "phm_rank": 1,
    "reduction_factor": 16,
    "residual_before_ln": True,
    "scaling": 1.0,
    "shared_W_phm": False,
    "shared_phm_rule": True
}

scores = list()
for seed in range(args.seed_runs):
    set_seed(seed)
    
    if "_pfeiffer_inv_" in model_name:
        # TODO: current hack to avoid the naming issue.
        assert False, "rename '_pfeiffer_inv_' to '_pfeiffer+inv_'"

    if "_pfeiffer_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_pfeiffer_{language}")

            model.add_adapter(f"wikiann-task-adapter")
            model.train_adapter(f"wikiann-task-adapter")
            model.active_adapters = Stack(pretrained_adapter_name, f"wikiann-task-adapter")
            print_model_trainable_layers(model)
            return model
    elif "_pfeiffer+inv_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_pfeiffer+inv_{language}")

            model.add_adapter(f"wikiann-task-adapter")
            model.train_adapter(f"wikiann-task-adapter")
            model.active_adapters = Stack(pretrained_adapter_name, f"wikiann-task-adapter")
            print_model_trainable_layers(model)
            return model
    elif "_aa_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_aa_{language}")

            model.add_adapter(f"wikiann-task-adapter")
            model.train_adapter(f"wikiann-task-adapter")
            model.active_adapters = Stack(pretrained_adapter_name, f"wikiann-task-adapter")
            print_model_trainable_layers(model)
            return model
    elif "_lora_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            with open(f"{model_name}/oscar_lora_{language}/adapter_config.json") as rf:
                lora_config_dict = json.load(rf)['config']

            config = ConfigUnion(
                LoRAConfig.from_dict(lora_config_dict),
                AdapterConfig.from_dict(task_config_dict)
            )

            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_lora_{language}")
            pretrained_adapter_modules = model.get_adapter(pretrained_adapter_name)
            cached_weights = {}
            for layer_i, v in pretrained_adapter_modules.items():
                for layer_name in v.keys():
                    for name, param in pretrained_adapter_modules[layer_i][layer_name].named_parameters():
                        cached_weights[f"{layer_i}-{layer_name}-{name}"] = param.data
            model.delete_adapter(pretrained_adapter_name)

            model.add_adapter("wikiann_union_adapter", config=config)
            model.train_adapter("wikiann_union_adapter")
            union_adapter_modules = model.get_adapter("wikiann_union_adapter")
            for layer_i, v in union_adapter_modules.items():
                for layer_name in v.keys():
                    for name, param in union_adapter_modules[layer_i][layer_name].named_parameters():
                        if f"{layer_i}-{layer_name}-{name}" in cached_weights:
                            # print(f"Replacing {layer_i}-{layer_name}-{name}")
                            param.data = cached_weights[f"{layer_i}-{layer_name}-{name}"]
                            param.requires_grad = False

            # pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_lora_{language}")
            # model.set_active_adapters(pretrained_adapter_name)
            # for name, param in model.get_adapter(pretrained_adapter_name)[0]['selfattn_lora'].named_parameters():
            #     print(name)
            #     print(param)

            # model.add_adapter(f"wikiann-task-adapter")
            # model.train_adapter(f"wikiann-task-adapter")
            print_model_trainable_layers(model)
            return model
    elif "_ia3_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            with open(f"{model_name}/oscar_ia3_{language}/adapter_config.json") as rf:
                ia3_config_dict = json.load(rf)['config']

            config = ConfigUnion(
                IA3Config.from_dict(ia3_config_dict),
                AdapterConfig.from_dict(task_config_dict)
            )

            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_ia3_{language}")
            pretrained_adapter_modules = model.get_adapter(pretrained_adapter_name)
            cached_weights = {}
            for layer_i, v in pretrained_adapter_modules.items():
                for layer_name in v.keys():
                    for name, param in pretrained_adapter_modules[layer_i][layer_name].named_parameters():
                        cached_weights[f"{layer_i}-{layer_name}-{name}"] = param.data
            model.delete_adapter(pretrained_adapter_name)

            model.add_adapter("wikiann_union_adapter", config=config)
            model.train_adapter("wikiann_union_adapter")
            union_adapter_modules = model.get_adapter("wikiann_union_adapter")
            for layer_i, v in union_adapter_modules.items():
                for layer_name in v.keys():
                    for name, param in union_adapter_modules[layer_i][layer_name].named_parameters():
                        if f"{layer_i}-{layer_name}-{name}" in cached_weights:
                            # print(f"Replacing {layer_i}-{layer_name}-{name}")
                            param.data = cached_weights[f"{layer_i}-{layer_name}-{name}"]
                            param.requires_grad = False
            print_model_trainable_layers(model)
            return model
    elif "_prefix_tuning_" in model_name or "_prompt_tuning_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            with open(f"{model_name}/oscar_prefix_tuning_{language}/adapter_config.json") as rf:
                prefix_tuning_config_dict = json.load(rf)['config']

            config = ConfigUnion(
                PrefixTuningConfig.from_dict(prefix_tuning_config_dict),
                AdapterConfig.from_dict(task_config_dict)
            )

            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_prefix_tuning_{language}")
            pretrained_adapter_modules = model.get_adapter(pretrained_adapter_name)
            cached_weights = {}
            for layer_i, v in pretrained_adapter_modules.items():
                for layer_name in v.keys():
                    for name, param in pretrained_adapter_modules[layer_i][layer_name].named_parameters():
                        cached_weights[f"{layer_i}-{layer_name}-{name}"] = param.data
            model.delete_adapter(pretrained_adapter_name)

            model.add_adapter("wikiann_union_adapter", config=config)
            model.train_adapter("wikiann_union_adapter")
            union_adapter_modules = model.get_adapter("wikiann_union_adapter")
            for layer_i, v in union_adapter_modules.items():
                for layer_name in v.keys():
                    for name, param in union_adapter_modules[layer_i][layer_name].named_parameters():
                        if f"{layer_i}-{layer_name}-{name}" in cached_weights:
                            # print(f"Replacing {layer_i}-{layer_name}-{name}")
                            param.data = cached_weights[f"{layer_i}-{layer_name}-{name}"]
                            param.requires_grad = False

            # model.add_adapter(f"wikiann-task-adapter")
            # model.train_adapter(f"wikiann-task-adapter")
            print_model_trainable_layers(model)
            
            return model
    else:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(model_name, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)

            model.add_adapter(f"wikiann-task-adapter")
            model.train_adapter(f"wikiann-task-adapter")
            print_model_trainable_layers(model)
            return model

    # model.freeze_model(True)
    
    # finetuning setting: https://aclanthology.org/2021.acl-long.172.pdf
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
        load_best_model_at_end=True, # will load the last saved **model** checkpoint, so will cause problem for adapters.
        metric_for_best_model='eval_overall_f1',
        local_rank=args.local_rank
    )
    if args.reproducible:
        trainer = AdapterTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
    else:
        model = model_init()
        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
    
    trainer.train()

    checkpoints_dir = list(pathlib.Path(f"{args.output_dir}/").glob("checkpoint-*"))
    checkpoints_dir.sort(key=lambda fp: int(fp.name.split('-')[-1]))
    with open(checkpoints_dir[-1] / "trainer_state.json") as rf:
        checkpoint = json.load(rf)['best_model_checkpoint']
        print("best_model_checkpoint:", checkpoint)

    if "_pfeiffer_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_pfeiffer_{language}")
            task_adapter_name = model.load_adapter(f"{checkpoint}/wikiann-task-adapter")

            model.active_adapters = Stack(pretrained_adapter_name, task_adapter_name)
            model.eval()

            print_model_trainable_layers(model)
            return model
    elif "_pfeiffer+inv_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_pfeiffer+inv_{language}")
            task_adapter_name = model.load_adapter(f"{checkpoint}/wikiann-task-adapter")

            model.active_adapters = Stack(pretrained_adapter_name, task_adapter_name)
            model.eval()

            print_model_trainable_layers(model)
            return model
    elif "_aa_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_aa_{language}")
            task_adapter_name = model.load_adapter(f"{checkpoint}/wikiann-task-adapter")

            model.active_adapters = Stack(pretrained_adapter_name, task_adapter_name)
            model.eval()

            print_model_trainable_layers(model)
            return model
    elif "_lora_" in model_name or \
         "_ia3_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            
            # pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_lora_{language}")
            # model.set_active_adapters(pretrained_adapter_name)
            task_adapter_name = model.load_adapter(f"{checkpoint}/wikiann_union_adapter")
            model.set_active_adapters(task_adapter_name)
            model.eval()

            print_model_trainable_layers(model)
            return model
    elif "_prefix_tuning_" in model_name or "_prompt_tuning_" in model_name:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(base_model, 
                                                                    pad_token_id=tokenizer.pad_token_id,
                                                                    cache_dir=args.cache_dir,
                                                                    num_labels=7)
            # pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_prefix_tuning_{language}")
            # model.set_active_adapters(pretrained_adapter_name)
            task_adapter_name = model.load_adapter(f"{checkpoint}/wikiann_union_adapter")
            model.set_active_adapters(task_adapter_name)
            model.eval()

            print_model_trainable_layers(model)
            return model
    else:
        def model_init():
            model = AutoModelForTokenClassification.from_pretrained(model_name, 
                                                                pad_token_id=tokenizer.pad_token_id,
                                                                cache_dir=args.cache_dir,
                                                                num_labels=7)
            task_adapter_name = model.load_adapter(f"{checkpoint}/wikiann-task-adapter")
            model.set_active_adapters(task_adapter_name)
            model.eval()

            print_model_trainable_layers(model)
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

    # clear CUDA memory: https://github.com/huggingface/transformers/issues/1742
    if not args.reproducible:
        del model
    del trainer
    del eval_trainer
    gc.collect()
    torch.cuda.empty_cache()


print("="*50)
print("Results")
print("="*50)
print("Model:", model_name)
print(scores)
print(np.mean(scores) * 100)
print(np.std(scores) * 100)
print("="*50)


# writing results to the model name.
with open(f"{model_name}/wikiann-{language}-results.txt", "w+") as wf:
    wf.write("="*50)
    wf.write('\n')
    wf.write("Results")
    wf.write('\n')
    wf.write("="*50)
    wf.write('\n')
    wf.write(f"Model: {model_name}")
    wf.write('\n')
    wf.write(f"{scores}")
    wf.write('\n')
    wf.write(f"{np.mean(scores) * 100:.2f}")
    wf.write('\n')
    wf.write(f"{np.std(scores) * 100:.2f}")
    wf.write('\n')
    wf.write("="*50)
