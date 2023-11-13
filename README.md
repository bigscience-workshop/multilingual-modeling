# README

This repository contains code for performing language adaptation of BLOOM-{560m,1b1,1b7,3b,7b1} to new unseen languages. Please refer to our ACL 2023 paper [BLOOM+1: Adding Language Support to BLOOM for Zero-Shot Prompting](https://aclanthology.org/2023.acl-long.653/).

Our implementations support the following features:
- finetuning new tokenizers and embedding layers to support new script of unseen languages.
- different language adaptation strategies for pretrained BLOOM model, including continued-pretraining and parameter-efficient finetuning such as adaptable adapters ([Moosavi et al., 2022](https://arxiv.org/abs/2205.01549)), BitFit ([Zaken et al., 2021](https://arxiv.org/abs/2106.10199)), (IA)^3 ([Liu et al., 2022](https://arxiv.org/abs/2205.05638)), LoRA ([Hu et al., 2021](https://arxiv.org/abs/2106.09685)), MAD-X ([Pfeiffer et al., 2020](https://aclanthology.org/2020.emnlp-main.617/)), and composible sparse finetuning ([Ansell et al., 2022](https://github.com/cambridgeltl/composable-sft)).
- task-finetuning with (English) task adapters on the following tasks: WikiANN (NER tagging), XLSum (abstractive summarization) and XNLI (natural language inference). This is an artefact that isn't used as evaluation benchmark in our BLOOM+1 paper

Zero-shot prompting on adapted language models, which is carried out on our [BLOOM+1](https://arxiv.org/abs/2212.09535) paper, is done with forked and modified EleutherAI's lm-eval-harness library. See branch [`bigscience-lm-adapt`](https://github.com/yongzx/lm-evaluation-harness/tree/bigscience-lm-adapt).


## Installation
1. Install the packages from [composable-sft](https://github.com/cambridgeltl/composable-sft). This is used for composable-SFT finetuning.
2. Install the packages from [rational_activations](https://github.com/ml-research/rational_activations). You would need to follow the [Other CUDA/PyTorch] section for installation. This is used for adaptable-adapters. 
3. Install the packages from this repo using `pip install -r requirements.txt`. 

If encounter error with the `import transformer`, uninstall transformers using the command `pip uninstall transformers` and rerun step 3 to reinstall `transformers` supported by `adapter-transformers` library.

## Language Adaptations

See the scripts in `lang_adapt/scripts/run_clm_*` as examples for the following language adaptation strategies:
- Adaptable adapters
- BitFit
- Continual Pretraining
- IA3
- LoRA
- MAD-X
- Pfeiffer
- Pretraining from Scratch
- Composable SFT


## Citation
```
@inproceedings{yong-etal-2023-bloom,
    title = "{BLOOM}+1: Adding Language Support to {BLOOM} for Zero-Shot Prompting",
    author = "Yong, Zheng Xin  and Schoelkopf, Hailey  and Muennighoff, Niklas  and Aji, Alham Fikri  and Adelani, David Ifeoluwa  and Almubarak, Khalid  and Bari, M Saiful  and Sutawika, Lintang  and Kasai, Jungo  and Baruwa, Ahmed  and Winata, Genta  and Biderman, Stella  and Raff, Edward  and Radev, Dragomir  and Nikoulina, Vassilina",
    editor = "Rogers, Anna  and Boyd-Graber, Jordan  and Okazaki, Naoaki",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.653",
    doi = "10.18653/v1/2023.acl-long.653",
    pages = "11682--11703",
}
```