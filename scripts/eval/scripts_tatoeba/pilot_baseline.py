import numpy as np
import collections
import json
import pathlib
import gc
from tqdm import tqdm

from transformers import set_seed

import torch

import sys
from loguru import logger
logger.remove()
logger.add(sys.stderr, format="{level} {level.icon} | [{time}] - {message}")

from datasets import load_metric
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, Seq2SeqTrainer, AdapterTrainer, Seq2SeqAdapterTrainer, Seq2SeqTrainingArguments
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from transformers.adapters.composition import Stack
from transformers import AdapterConfig, LoRAConfig, PrefixTuningConfig, ConfigUnion, IA3Config


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lang_pairs", type=str)
parser.add_argument("--cache_dir", type=str, default="/users/zyong2/data/zyong2/huggingface")
parser.add_argument("--output_dir", type=str)
parser.add_argument("--tokenizer", type=str)
parser.add_argument("--model_name", type=str)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--base_model", type=str, default="bigscience/bloom-1b3")
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument("--reproducible", action="store_true")
parser.add_argument("--seed_runs", type=int, default=3)
parser.add_argument("--num_pairs", type=int, default=200)
parser.add_argument("--device", type=str)
parser.add_argument("--use_checkpoints", action="store_true")
args = parser.parse_args()

language1, language2 = args.lang_pairs.split("-")
all_datasets = []
for i in range(args.seed_runs):
    start = i * args.num_pairs
    end = (i + 1) * args.num_pairs
    # print("loaded dataset indexes:", start, "-", end)

    train_dataset = load_dataset("tatoeba", lang1=language1, lang2=language2, cache_dir=args.cache_dir, split=f"train[{start}:{end}]")
    language1_dataset = list()
    language2_dataset = list()
    for i in tqdm(range(len(train_dataset)), desc=f"loading dataset (n={args.num_pairs})"):
        language1_dataset.append(train_dataset['translation'][i][language1])
        language2_dataset.append(train_dataset['translation'][i][language2])
    all_datasets.append((language1_dataset, language2_dataset))

tok = args.tokenizer
model_name = args.model_name
base_model = args.base_model

tokenizer = AutoTokenizer.from_pretrained(tok, cache_dir=args.cache_dir, add_prefix_space=True)
if not tokenizer.pad_token:
    if 'mGPT' in model_name:
        tokenizer.pad_token = '<pad>' #mGPT 


def nxn_cos_sim(A, B, dim=1, eps=1e-8):
    numerator = A @ B.T
    A_l2 = torch.mul(A, A).sum(axis=dim)
    B_l2 = torch.mul(B, B).sum(axis=dim)
    denominator = torch.max(torch.sqrt(torch.outer(A_l2, B_l2)), torch.tensor(eps))
    return torch.div(numerator, denominator)

def print_model_trainable_layers(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"🥶 Frozen layer '{name}'")
        else:
            print(f"🚀 Trainable layer '{name}'")
        
        # print(param)
    
    print(model)
    
if "_pfeiffer_inv_" in model_name:
    # TODO: current hack to avoid the naming issue.
    assert False, "rename '_pfeiffer_inv_' to '_pfeiffer+inv_'"

if "_pfeiffer_" in model_name:
    def model_init():
        model = AutoModel.from_pretrained(base_model, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            cache_dir=args.cache_dir)
        pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_pfeiffer_{language}")
        model.set_active_adapters(pretrained_adapter_name)
        # print_model_trainable_layers(model)
        return model
elif "_pfeiffer+inv_" in model_name:
    def model_init():
        model = AutoModel.from_pretrained(base_model, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            cache_dir=args.cache_dir)
        pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_pfeiffer+inv_{language}")
        model.set_active_adapters(pretrained_adapter_name)
        # print_model_trainable_layers(model)
        return model
elif "_aa_" in model_name:
    def model_init():
        model = AutoModelfrom_pretrained(base_model, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            cache_dir=args.cache_dir)
        pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_aa_{language}")
        model.set_active_adapters(pretrained_adapter_name)
        # print_model_trainable_layers(model)
        return model
elif "_lora_" in model_name:
    def model_init():
        model = AutoModel.from_pretrained(base_model, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            cache_dir=args.cache_dir)
        pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_lora_{language}")
        model.set_active_adapters(pretrained_adapter_name)
        # print_model_trainable_layers(model)
        return model
elif "_ia3_" in model_name:
    def model_init():
        model = AutoModel.from_pretrained(base_model, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            cache_dir=args.cache_dir)

        pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_ia3_{language}")
        model.set_active_adapters(pretrained_adapter_name)
        # print_model_trainable_layers(model)
        return model
elif "_prefix_tuning_" in model_name or "_prompt_tuning_" in model_name:
    def model_init():
        model = AutoModel.from_pretrained(base_model, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            cache_dir=args.cache_dir)
        pretrained_adapter_name = model.load_adapter(f"{model_name}/oscar_prefix_tuning_{language}")
        model.set_active_adapters(pretrained_adapter_name)
        # print_model_trainable_layers(model)
        return model
else:
    def model_init():
        model = AutoModel.from_pretrained(model_name, 
                                            pad_token_id=tokenizer.pad_token_id,
                                            cache_dir=args.cache_dir)
        # print_model_trainable_layers(model)
        return model

def evaluate_on_tatoeba(model):
    language = language1 if language1 != "en" else language2 # language for adapters
    scores = list()

    for i in tqdm(range(args.seed_runs), desc="seeded runs"):
        language1_dataset, language2_dataset = all_datasets[i]
        sentence_embs1 = list()
        sentence_embs2 = list()

        # get all the sentence representations
        # avoid padding
        for sent in tqdm(language1_dataset, desc=f"Going through {language1}, compute embeddings"):
            x = tokenizer(sent, return_tensors="pt").input_ids.to(model.device)
            output = model(x)
            hidden_states = output.last_hidden_state.detach()
            sentence_emb = torch.mean(hidden_states[0], dim=0).tolist()
            sentence_embs1.append(sentence_emb)

        for sent in tqdm(language2_dataset, desc=f"Going through {language2}, compute embeddings"):
            x = tokenizer(sent, return_tensors="pt").input_ids.to(model.device)
            output = model(x)
            hidden_states = output.last_hidden_state.detach()
            sentence_emb = torch.mean(hidden_states[0], dim=0).tolist()
            sentence_embs2.append(sentence_emb)

        # calculate nxn cosine similarity
        embs1 = torch.Tensor(sentence_embs1)
        embs2 = torch.Tensor(sentence_embs2)
        sim = nxn_cos_sim(embs1, embs2)

        # calculate accuracy score
        labels = torch.argmax(sim, dim=1)
        gold_labels = torch.arange(len(labels))
        scores.append(torch.sum(labels == gold_labels) / len(labels))
    
    return scores


model = model_init()
model = model.to(args.device)
# print("Model's device:", model.device)
model = model.eval()

##### RESULTS
scores = evaluate_on_tatoeba(model)
print("="*50)
print(f"Tatoeba Results ({args.num_pairs} pairs of {args.lang_pairs})")
print("="*50)
print("Model:", model_name)
print(scores)
print(f"{np.mean(scores) * 100:.2f} ± {np.std(scores) * 100:.2f}")
print("="*50)


    # # writing results to the model name.
    # with open(f"{model_name}/tatoeba-{language1}-{language2}-results.txt", "w+") as wf:
    #     wf.write("="*50)
    #     wf.write('\n')
    #     wf.write(f"Tatoeba Results ({args.num_pairs} pairs of {args.lang_pairs})")
    #     wf.write('\n')
    #     wf.write("="*50)
    #     wf.write('\n')
    #     wf.write(f"Model: {model_name}")
    #     wf.write('\n')
    #     wf.write(f"{scores}")
    #     wf.write('\n')
    #     wf.write(f"{np.mean(scores) * 100:.2f} ± {np.std(scores) * 100:.2f}")
    #     wf.write('\n')
    #     wf.write("="*50)
