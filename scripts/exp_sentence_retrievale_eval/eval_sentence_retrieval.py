import logging
import argparse
import os
from datasets import load_dataset
from collections import namedtuple
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os.path
import sys
from loguru import logger
import random
logger.remove()
logger.add(sys.stderr, format="{level} {level.icon} | [{time}] - {message}")


# parser
parser = argparse.ArgumentParser()
parser.add_argument("output_dir")
parser.add_argument("--pretrained_model", default="bert-base-multilingual-cased")
parser.add_argument("--tokenizer", default="bert-base-multilingual-cased")
parser.add_argument("--dataset", default="ted_multi")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
ted_lngs = ['am', 'ar', 'bn', 'ca', 'en', 'es', 'fr', 'hi', 'id', 'ja', 'pt', 'zh-cn', 'zh-tw', 'pt-br']
flores_lng = ["amh", "bos", "cat", "eng", "spa", "fra", "hin", "ind", "jpn", "por", "swh", "vie", "urd"]
bs_languages = ["id", "eu", "vi", "zh", "ur", "es", "ca", "pt", "fr", "en", "hi", "ar", "bn"]
lngcode_map = {"am":"amh", "bn":"bos", "ca":"cat", "en":"eng", "es":"spa", "fr": "fra", "hi": "hin", "id": "ind", "ja": "jpn", "pt": "por", "ur":"urd", "vi":"vie" }


print("Arguments: ========")
print(args)


def load_dataset_(args):
    if args.dataset == "ted_multi":
        return load_dataset_ted(args)
    if args.dataset == "flores":
        return load_dataset_flores(args)


def load_dataset_flores_for_lng(args, lng):
    dataset = load_dataset("gsarti/flores_101", lngcode_map[lng])['dev']
    return dataset

def load_dataset_flores(args):
    dataset = {}
    for lng in bs_languages:
        if lng in lngcode_map:
            load_dataset_flores_for_lng(args, lng)
    return dataset
            
def load_dataset_ted(args):
    dataset = load_dataset("ted_multi")['validation']
    return dataset

def get_talks(dataset, nb_talks):
    talk_names = []
    for t in dataset['talk_name']:
        if len(talk_names) < nb_talks and not t in talk_names:
            talk_names.append(t)


    print([(t1, len([t for t in dataset['talk_name'] if t == t1])) for t1 in talk_names])
    return talk_names

def load_model(args):
    if "xlm" in args.pretrained_model or "bert" in args.pretrained_model:
        model = AutoModelForMaskedLM.from_pretrained(args.pretrained_model)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.pretrained_model)
    model.config.output_hidden_states=True
    return model.to(args.device)

Sample = namedtuple(
    "Sample",
    ("id", "hidden_state")
)

def load_from_file(fname):
    return torch.load(fname)


def get_hidden_states(args, model):
    if args.dataset == "ted_multi":
        dataset = load_dataset_(args)
        nb_talks = 2
        talks = get_talks(dataset, nb_talks)

        emb = get_hidden_states_for_talks(dataset, model, talks, args.pretrained_model)

        outname = f"{args.output_dir}/{args.pretrained_model.replace('/','-')}-talks-valid-{len(talks)}"

    elif args.dataset == "flores":
        nb_samples = 200
        emb = get_hidden_states_for_flores(args, model, args.pretrained_model, nb_samples = nb_samples)
        outname = f"{args.output_dir}/{args.pretrained_model.replace('/','-')}-flores-{nb_samples}"

    retrieval_acc = {}
    nb_states = model.config.num_hidden_layers
    fig, ax = plt.subplots(1, int(nb_states/step), figsize=(12*int(nb_states/step), 10))


    with open(f"{outname}.log", 'w') as fout:
        for state in range(0, nb_states, step):
            plot_retrieval_acc(state, emb, ax[int(state/step)], fout)

    fig.tight_layout()
    plt.savefig(f'{outname}-heatmap.png')


def get_hidden_states_for_flores(args, model, mname, nb_samples=50):
    emb = {}
    hidden_state_size = model.config.num_hidden_layers
    for lng in bs_languages:        
        if lng in lngcode_map:
            fname = f"{args.output_dir}/flores-{lng}-{nb_samples}-{mname.replace('/','-')}.pt"
            if os.path.isfile(fname):
                emb[lng] = load_from_file(fname)
            else:
                dataset = load_dataset_flores_for_lng(args, lng)
                emb[lng] = {}
                for state in range(hidden_state_size):
                    emb[lng][state] = []
                for i, sid in enumerate(dataset['id'][:nb_samples]):
                    t = dataset['sentence'][i]
                    x = tokenizer(t,  return_tensors="pt").input_ids.to(model.device)
                    out = model(x)
                    for state in range(hidden_state_size):
                        hs = torch.mean(out.hidden_states[state][0][1:-1], dim=0).detach()
                        emb[lng][state].append(Sample(sid, hs))
                torch.save(emb[lng], fname)
    return emb


def get_hidden_states_for_talks(dataset, model, talks, mname):
    emb = {}
    hidden_state_size = model.config.num_hidden_layers
    fname = f"{args.output_dir}/ted_multi-{mname.replace('/','-')}-ted_multi-{len(talks)}.pt"
    if os.path.isfile(fname):
        emb = load_from_file(fname)
        return emb
    for sid, sample in enumerate(dataset):
        if sample['talk_name'] in talks:
            tsample = sample['translations']
            for i, lng in enumerate(tsample['language']):
                if lng in bs_languages:
                    t = tsample['translation'][i]            
                    x = tokenizer(t,  return_tensors="pt").input_ids.to(model.device)
                    if not lng in emb:
                        emb[lng] = {}
                        for state in range(hidden_state_size):
                            emb[lng][state] = []
                    out = model(x)
                    for state in range(hidden_state_size):
                        hs = torch.mean(out.hidden_states[state][0], dim=0).detach()
                        emb[lng][state].append(Sample(sid, hs))
    torch.save(emb, fname)
    return emb


def compute_sent_retrieval_acc(lng1, lng2, emb, state, out):
    cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
    E1 = torch.stack([s[1] for s in emb[lng1][state]])
    E2 = torch.stack([s[1] for s in emb[lng2][state]])
    #cos_matrix = [[cos(E2[i],E1[j]) for i in range(E2.shape[0]) ] for j in range(E1.shape[0])]    
    match = 0
    intersection_ids = set([emb[lng1][state][i][0] for i in range(E1.shape[0])]).intersection(
        set([emb[lng2][state][i][0] for i in range(E2.shape[0])])
    )
    if len(intersection_ids)>0:
        random_acc = 1/len(intersection_ids)
        for i in range(E1.shape[0]):
            if emb[lng1][state][i][0] in intersection_ids:
                cos_sim = [cos(E2[j], E1[i]) for j in range(E2.shape[0])]
                best_match = torch.argmax(torch.stack(cos_sim))
                if emb[lng2][state][best_match][0] == emb[lng1][state][i][0]:
                    match +=1
        acc = match/len(intersection_ids)
        out.write(f"{lng1}-{lng2} = {acc} (random {random_acc} )\n")
        return acc, len(intersection_ids)
    else:
        return 0, 0

def plot_retrieval_acc(state, emb, ax, out):
    cmap="RdYlBu"
    mean_per_state = 0
    for lng1 in emb:
        if not lng1 in retrieval_acc:
            retrieval_acc[lng1] = {}
        for lng2 in emb:
            lng2_chance = 1.0/len(emb[lng2][0])
            #if not lng1 == lng2:
            acc, random_acc = compute_sent_retrieval_acc(lng1, lng2, emb, state, out)
            retrieval_acc[lng1][lng2] = acc
            #retrieval_acc[lng1]["random"] = lng2_chance
        mean_acc = np.mean([v for v in retrieval_acc[lng1].values()])
        out.write(f"ACC per {lng1}, layer {state} = {mean_acc} \n" ) 
        mean_per_state +=mean_acc
    mean_per_state = mean_per_state/len(emb.keys())
    out.write(f"ACC overall, layer {state} = {mean_per_state}\n" ) 
    m_res = pd.DataFrame(retrieval_acc)
    m_res.columns=emb.keys()
    m_res.index=emb.keys()#[e for e in emb.keys()]+["random"]
    ax.set_title(f"state {state}")
    sns.heatmap(m_res, ax=ax, annot=False, vmin=0, vmax=1.0, center=0, cmap=cmap)



lngs2consider = ['am', 'ar', 'bn', 'ca', 'en', 'es', 'fr', 'hi', 'id', 'ja', 'pt', 'zh-cn', 'zh-tw', 'pt-br']
samples = 10
model = load_model(args)
retrieval_acc = {}
step=1
get_hidden_states(args, model)
