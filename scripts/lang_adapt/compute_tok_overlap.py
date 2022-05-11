import sys
import json
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from collections import defaultdict
import math
import argparse
import matplotlib.pyplot as plt

def get_en_tokenizer():
    en_tok = AutoTokenizer.from_pretrained('/tmp-network/user/vnikouli/Projects/bigscience/multilingual-modeling/scripts/exp-009/tr5b-1B3-multilingual-alpha-checkpoints/') 
    return en_tok

def getdata(lng):
    flores_path="/tmp-network/user/vnikouli/Projects/NLE-NMT/data/test_sets/"
    with open(f'{flores_path}/FLORES-valid.{lng}') as f:
        dataset = f.readlines()
    return dataset

def gettokens(tok, dataset):
    from collections import defaultdict
    seq_lengths = []
    toks_occ = defaultdict(int)
    for i,l in enumerate(dataset):        
        toks = tok.tokenize(l.strip())
        seq_lengths.append(len(toks))
        toks_occ.update({t:toks_occ[t]+1 for t in toks })
    return np.array(seq_lengths), toks_occ



def plot_histogram(tokoccs, name, ax, nb_bins):
    ax.hist(tokoccs, nb_bins, histtype='bar', label=name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--tokenizers', type=str,  nargs='+',
                    help='an integer for the accumulator')
    parser.add_argument('--plot_name', type=str, default="stats_plot")
    args = parser.parse_args()
    lng = args.lang
    tokenizers = args.tokenizers
    vocabs = {}
    dataset=getdata(lng)
    en_dataset = getdata("en")
    seq_lengths = {}
    tok_occs = {}
    en_tok = get_en_tokenizer()
    sl, to = gettokens(en_tok, en_dataset)
    seq_lengths['en'] = sl

    for t in tokenizers:
        tok = AutoTokenizer.from_pretrained(t)
        sl, to = gettokens(tok, dataset)
        seq_lengths[t] = sl
        tok_occs[t] = to
        with open(f'{t}/vocab.json') as jsonFile:
            vocab = json.load(jsonFile)
            vocabs[t] = set(vocab.keys())


    print("Print tokenization stats")
    print("===============================")
    fig, ax = plt.subplots(1, 4, figsize=(40, 10))
    for t in tokenizers:
        print(f'Tokenizer {t}, avg tokenized seq length: {np.mean(seq_lengths[t])} (shorter sequences are better)')
        #we want to decompose sentence in {lng} in approximately the same nb of tokens as in English hoping that it will favour knowledge transfer
        x = seq_lengths[t]/seq_lengths["en"]
        print(f'Tokenizer {t}, avg ratio with En tokenized sentence length: {np.mean(x)}+/- {np.std(x)}') 
        baseline_overlap = vocabs[t].intersection(set(en_tok.vocab.keys()))
        print(f"Overlap with original tokenizer vocab : {len(baseline_overlap)} ")
        print(f"Overlap between new tokenizer vocab and obtained tokenswith original tokenizer vocab : {len(baseline_overlap)} ")



    print("Do plotting")
    fig, ax = plt.subplots(1, 4, figsize=(40, 10))
    ax[0].set_title("Token occ distribution")
    plot_histogram([[math.log(v) for v in tok_occs[t].values()] for t in tokenizers], tokenizers, ax[0], 10)
    ax[1].set_title("Seq length distribution")    
    plot_histogram([seq_lengths[t] for t in tokenizers], tokenizers, ax[1], 10)
    ax[2].set_title("Diff wtih en seq length distribution")    
    plot_histogram([seq_lengths[t]/seq_lengths["en"] for t in tokenizers], tokenizers, ax[2], 10)
    ax[3].set_title("Tok length distribution")
    plot_histogram([[len(v) for v in vocabs[t] for i in range(tok_occs[t][v])] for t in tokenizers], tokenizers, ax[3], 10)
    ax[1].legend()
    fig.savefig(f"{args.plot_name}.png")    


