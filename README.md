# README

## Installation
1. Install the packages from [composable-sft](https://github.com/cambridgeltl/composable-sft).
2. Install the packages from [rational_activations](https://github.com/ml-research/rational_activations). You would need to follow the [Other CUDA/PyTorch] section for installation. 
3. Uninstall transformers using the command `pip uninstall transformers`. 
4. Install the packages from this repo using `pip install -r requirements.txt`. 

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

## WikiANN

See the scripts in `scripts/eval/scripts_wikiann/pilot_*_.sh` as examples for evaluating the adapted models on WikiANN.

## Plots

- BLOOMZ-7b1-xP3RU: [Colab](https://colab.research.google.com/drive/1GYjuR2hH3rYNh4vMeheV12Q-5JUP7RxA?usp=sharing)
