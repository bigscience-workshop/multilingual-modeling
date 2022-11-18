"""
Source: https://github.com/Adapter-Hub/adapter-transformers/blob/master/examples/language-modeling/run_clm.py
"""

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import pathlib 

import datasets
from datasets import load_dataset

import transformers
from transformers import EarlyStoppingCallback

import transformers.adapters.composition as ac
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AdapterTrainer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    MultiLingAdapterArguments,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.adapters.configuration import AdapterConfig
from transformers.adapters import PrefixTuningConfig, LoRAConfig, IA3Config, CompacterPlusPlusConfig, PfeifferInvConfig, ConfigUnion

from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

from sft import (
    LotteryTicketSparseFineTuner,
)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.11.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


### Implementations for FISH mask
class SparseUpdateTrainer(Trainer):
    """
    Trainer for FISH Mask
    """
    def __init__(self, *args, mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask = mask

    def training_step(self, *args, **kwargs):
        loss = super().training_step(*args, **kwargs)

        # mask out the gradients
        for name, params in self.model.named_parameters():
            if not params.requires_grad:
                continue

            device = params.device
            self.mask[name] = self.mask[name].to(device)

            params.grad.data.copy_(params.grad.data * self.mask[name].data)

        return loss

def calculate_the_importance_label(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        loss = return_dicts["loss"]

        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad == False:
                continue
            gradients_dict[name] += grad_method(param.grad).data
        
        model.zero_grad()

    return gradients_dict


def calculate_the_importance_expect(model, data_loader, num_samples, cuda_device, grad_type):
    """
    Args:
        grad_type: (square or absolute) 
    """
    gradients_dict = {}

    for name, param in model.named_parameters():
        gradients_dict[name] = torch.zeros_like(param).to(cuda_device)

    if grad_type == "absolute":
        grad_method = torch.abs
    elif grad_type == "square":
        grad_method = torch.square

    for idx, inputs in enumerate(data_loader):
        if idx >= num_samples:
            break

        # print(idx)

        inputs.pop("idx", None)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(cuda_device)

        return_dicts = model(**inputs)

        logits = return_dicts["logits"]

        log_probs = torch.nn.functional.log_softmax(logits, -1)
        probs = torch.nn.functional.softmax(logits, -1)

        for b in range(logits.shape[0]):
            for i in range(logits.shape[1]):
                loss = - log_probs[b, i]
                loss.backward(retain_graph=True)

                prob = probs[b, i]

                for name, param in model.named_parameters():
                    gradients_dict[name] += (prob * grad_method(param.grad)).data

                model.zero_grad()

    return gradients_dict

def create_fish_mask_gradient(model, train_dataset, data_collator, num_samples, keep_ratio, keep_num, sample_type, grad_type):
    original_device = list(model.parameters())[0].device
    cuda_device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(cuda_device)

    from torch.utils.data import DataLoader
    data_loader = DataLoader(
        train_dataset,
        batch_size=1,
        collate_fn=data_collator,
        shuffle=True
    )

    if sample_type == "label":
        importance_method = calculate_the_importance_label
    elif sample_type == "expect":
        importance_method = calculate_the_importance_expect
    else:
        raise NotImplementedError

    gradients = importance_method(model, data_loader, num_samples, cuda_device, grad_type)

    # add sizes and aggregate tensors
    sizes = {}
    tensors = []

    classifier_size = 0 # for BLOOM language adaptation, classifier_size == 0.
    all_params_size = 0

    classifier_mask_dict = {}

    for k, v in gradients.items():
        # don't count classifier layer, they should be all trainable
        if "classifier" in k:
            classifier_size += torch.prod(torch.tensor(v.shape)).item()
            classifier_mask_dict[k] = torch.ones_like(v).to(original_device)
        else:
            sizes[k] = v.shape
            tensors.append(v.view(-1))

        all_params_size += torch.prod(torch.tensor(v.shape)).item()

    tensors = torch.cat(tensors, 0)

    if keep_num == 0:
        keep_num = int(all_params_size * keep_ratio) - classifier_size

    assert keep_num > 0

    tensors = tensors.to("cpu")  # transfer to CPU to avoid CUDA memory error.
    top_pos = torch.topk(tensors, keep_num)[1]

    masks = torch.zeros_like(tensors, device=cuda_device)

    masks[top_pos] = 1

    assert masks.long().sum() == len(top_pos)

    mask_dict = {}

    now_idx = 0
    for k, v in sizes.items():
        end_idx = now_idx + torch.prod(torch.tensor(v))
        mask_dict[k] = masks[now_idx: end_idx].reshape(v).to(original_device)
        now_idx = end_idx

    assert now_idx == len(masks)

    # Add the classifier's mask to mask_dict
    mask_dict.update(classifier_mask_dict)

    model.to(original_device)

    # Print the parameters for checking
    classifier_size = 0
    all_params_size = 0
    pretrain_weight_size = 0
    
    for k, v in mask_dict.items():
        if "classifier" in k:
            classifier_size += (v == 1).sum().item()
        else:
            pretrain_weight_size += (v == 1).sum().item()

        all_params_size += torch.prod(torch.tensor(v.shape)).item()
    
    print(pretrain_weight_size, classifier_size, all_params_size)
    # print(f"trainable parameters: {(pretrain_weight_size + classifier_size) / all_params_size * 100} %")
    print(f"[FISH] Total trainable parameters: {pretrain_weight_size + classifier_size}")

    return mask_dict

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
            "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: str = field(
        default="",
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    tie_word_embeddings: int = field(
        default=1,
        metadata={"help": "1: tie. 0: decoupled embedding"},
    )
    lang_adapt_strategies: str = field(
        default=None,
        metadata={"help": "language adaptation strategies"},
    )
    embedding_strategies: str = field(
        default="",
        metadata={"help": "choose one of the two strategies - 'replace', 'extend', 'overlap-replace'"},
    )
    adapter_placement: str = field(
        default="all", 
        metadata={"help": "list of layers where to place the adapters: all: use all layers, '17,24': list layers id separated by ','"},
    )
    
    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_lang: Optional[str] = field(
        default=None, metadata={"help": "The language of the dataset"}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Optional input sequence length after tokenization. "
            "The training dataset will be truncated in block of this size for training. "
            "Default to the model max input length for single sentence inputs (take into account special tokens)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."

@dataclass
class ParamEfficientArguments(MultiLingAdapterArguments):
    """
    Arguments pertaining to other parameter efficient techniques such as (LoRA, BitFit, etc.)
    """


    # lora
    selfattn_lora: bool = field(
        default=True,
        metadata={"help": "If True, add LoRA to the self-attention weights of a model. Defaults to True."},
    )
    intermediate_lora: bool = field(
        default=False,
        metadata={"help": "If True, add LoRA to the intermediate MLP weights of a model. Defaults to False."},
    )
    output_lora: bool = field(
        default=False,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    r_lora: Optional[int] = field(
        default=8,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    alpha_lora: Optional[int] = field(
        default=8,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    dropout_lora: Optional[float] = field(
        default=0.0,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    init_weights_lora: Optional[str] = field(
        default='lora',
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    composition_mode_lora: Optional[str] = field(
        default='add'
    )
    attn_matrices_lora: List[str] = field(
        default_factory=lambda: ["q", "v"]
    )

    # ia3
    r_ia3: Optional[int] = field(
        default=1,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    alpha_ia3: Optional[int] = field(
        default=1,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    dropout_ia3: Optional[float] = field(
        default=0.0,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    init_weights_ia3: Optional[str] = field(
        default='ia3',
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    composition_mode_ia3: Optional[str] = field(
        default='scale'
    )
    attn_matrices_ia3: List[str] = field(default_factory=lambda: ["k", "v"])

    # prefix tuning
    encoder_prefix: bool = field(
        default=True,
        metadata={"help": "If True, add prefixes to the encoder of an encoder-decoder model."},
    )
    cross_prefix: bool = field(
        default=True,
        metadata={"help": "If True, add prefixes to the cross attention of an encoder-decoder model."},
    )
    flat: bool = field(
        default=False,
        metadata={"help": "If True, train the prefix parameters directly. Otherwise, reparametrize using a bottleneck MLP."},
    )
    prefix_length: int = field(
        default=30,
        metadata={"help": "The length of prefix."},
    )
    bottleneck_size: Optional[int] = field(
        default=512,
        metadata={"help": "If flat=False, the size of the bottleneck MLP."},
    )
    non_linearity: Optional[str] = field(
        default="tanh",
        metadata={"help": "If flat=False, the non-linearity used in the bottleneck MLP."},
    )
    prefix_dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "If True, add LoRA to the output MLP weights of a model. Defaults to False."},
    )
    # leave_out: Optional[List]


    ### composable SFT unique args ###
    train_sft: bool = field(
        default=False, metadata={"help": "Whether to train sparse fine-tuning."}
    )
    full_ft_max_steps_per_iteration: Optional[int] = field(
        default=5000,
        metadata={
            "help": "Maximum number of steps per parameter selection iteration during full fine-tuning."},
    )
    sparse_ft_max_steps_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum of steps per sparse fine-tuning iteration. Overriden by `--max_steps` if set."},
    )
    full_ft_min_steps_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Minimum number of steps per parameter selection iteration during full fine-tuning."
        },
    )
    sparse_ft_min_steps_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Minimum of steps per parameter selection iteration during sparse fine-tuning."
        },
    )
    full_ft_max_epochs_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of epochs per parameter selection iteration during full fine-tuning."
        },
    )
    sparse_ft_max_epochs_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of epochs per parameter selection iteration during sparse fine-tuning."
        },
    )
    n_ft_iterations: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of parameter selection iterations during fine-tuning."},
    )
    ft_params_proportion: Optional[float] = field(
        default=None,
        metadata={
            "help": "The proportion of model parameters for which to learn non-zero differences during fine-tuning.\
                Will override `ft_params_num` if both are set."},
    )
    ft_params_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of model parameters for which to learn non-zero differences during fine-tuning. \
            Defaults to a number equivalent to the number of adapter (reduction factor 16) parameters."},
    )
    freeze_head: bool = field(
        default=False,
        metadata={"help": "Whether to freeze language modeling head."},
    )
    untie_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to untie input and output embeddings."},
    )
    freeze_layer_norm: bool = field(
        default=True,
        metadata={"help": "Whether to freeze layer normalisation parameters."},
    ) # changed from False to True
    full_l1_reg: Optional[float] = field(
        default=0.1, metadata={"help": "Coefficient of L1 regularisation during full fine-tuning."}
    ) # changed from 0.0 to 0.1
    sparse_l1_reg: Optional[float] = field(
        default=0.1, metadata={"help": "Coefficient of L1 regularisation during sparse fine-tuning."}
    ) # changed from 0.0 to 0.1
    apply_reg_to_sparse_only: bool = field(
        default=False,
        metadata={
            "help": "If true, only applies regularisation to those parameters which are eligible for sparse fine-tuning."
        },
    )
    sparse_ft_method: Optional[str] = field(
        default='LotteryTicket',
        metadata={"help": 'Sparse fine-tuning method. Can be LotteryTicket or Random.'},
    )

    # FISH mask
    fish_num_samples: int = field(
        default=1024,
        metadata={"help": "The number of samples to compute parameters importance"}
    )
    fish_keep_ratio: float = field(
        default=0.005,
        metadata={"help": "The trainable parameters to total parameters."}
    )
    fish_keep_num: int = field(
        default=0,
        metadata={"help": "Number of trainable paramers. Override fish_keep_ratio"}
    )
    fish_mask_method: str = field(
        default="label-square",
        metadata={"help": "The method to select trainable parameters. Format: sample_type-grad_type, \
                   where sample_type in \{label, expect\}, and grad_type in \{absolute, square\}"}
    )
    normal_training: bool = field(
        default=False,
        metadata={"help": "Whether to use typical BERT training method."}
    )
    mask_path: str = field(
        default="",
        metadata={"help": "The path for existing mask."}
    )
    data_split_path: str = field(
        default="",
        metadata={"help": "The path for existing training data indices."}
    )


def load_tokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        print(f"✅ load tokenizer from: {model_args.tokenizer_name}")
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        print(f"✅ load tokenizer from: {model_args.model_name_or_path}")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer



def load_data(data_args, model_args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None and data_args.train_file is not None:
        # Create a dataset from the provided file
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)

        # Downloading and loading a dataset from the hub.
        raw_hf_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        raw_datasets['train'] = datasets.concatenate_datasets(raw_datasets['train'], raw_hf_datasets['train']).shuffle(seed=42)

    elif data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )

    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir, **dataset_args)

    if "validation" not in raw_datasets.keys():
        if data_args.max_eval_samples is not None and data_args.max_train_samples is not None:                
            raw_datasets = raw_datasets['train'].train_test_split(train_size = data_args.max_train_samples, test_size = data_args.max_eval_samples)
        elif data_args.max_eval_samples is not None :                
            raw_datasets = raw_datasets['train'].train_test_split(test_size = data_args.max_eval_samples)
        else:
            raw_datasets = raw_datasets['train'].train_test_split(test_size = data_args.validation_split_percentage/100.0)
        raw_datasets['validation'] = raw_datasets['test']
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if data_args.max_train_samples is not None and len(raw_datasets['train']) > data_args.max_train_samples:
        # FIXME: currently assume the loaded checkpoint is trained with the first data_args.max_train_samples number of samples
        #raw_datasets["train"] = raw_datasets["train"].filter(lambda example, indice: indice < data_args.max_train_samples, with_indices=True)
        print(raw_datasets["train"])
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
        print(raw_datasets["train"])

    if data_args.max_eval_samples is not None and len(raw_datasets['validation']) > data_args.max_eval_samples:
        raw_datasets["validation"] = raw_datasets["validation"].select(range(data_args.max_eval_samples))

    print("✅ Loaded Raw Dataset:")
    print(raw_datasets)
    return raw_datasets

def load_model(model_args, tokenizer):
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": model_args.use_auth_token,
        "tie_word_embeddings": bool(model_args.tie_word_embeddings),
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=model_args.use_auth_token,
        )
        print(f"✅ load model from: {model_args.model_name_or_path}")
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
        print(f"✅ load model from config: ")
        print(config)

    #TODO: remap embedding parameters

    return model

def preprocess_data(training_args, data_args, model_args, tokenizer):
    with training_args.main_process_first(desc="dataset map tokenization"):
        # cache tokenized data
        base_cache_dir = f"{model_args.cache_dir}/{data_args.dataset_name}/{data_args.dataset_config_name}"
        saved_tokenized_datasets_fp = pathlib.Path(f"{base_cache_dir}/tokenized_data_{data_args.max_train_samples}train_{data_args.max_eval_samples}eval_{len(tokenizer)}vocab.pt")

        if not data_args.overwrite_cache and saved_tokenized_datasets_fp.exists() and saved_tokenized_datasets_fp.is_file():
            tokenized_datasets = torch.load(str(saved_tokenized_datasets_fp))
            logger.info(f"✅ loaded tokenized_data from {saved_tokenized_datasets_fp}")
        else:
            raw_datasets = load_data(data_args, model_args)
            assert len(raw_datasets['train']) == data_args.max_train_samples
            assert len(raw_datasets['validation']) == data_args.max_eval_samples
            assert len(raw_datasets['test']) == data_args.max_eval_samples
            print(f"✅ Sanity check: loaded raw datasets have {data_args.max_train_samples} training samples and {data_args.max_eval_samples} eval samples")
                                                      
            # First we tokenize all the texts.
            if training_args.do_train:
                column_names = raw_datasets["train"].column_names
            else:
                column_names = raw_datasets["validation"].column_names

            text_column_name = "text" if "text" in column_names else column_names[0]
            # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
            tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

            def tokenize_function(examples):
                with CaptureLogger(tok_logger) as cl:
                    output = tokenizer(examples[text_column_name])
                    # clm input could be much much longer than block_size
                    if "Token indices sequence length is longer than the" in cl.out:
                        tok_logger.warning(
                            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits before being passed to the model."
                        )
                    return output
            
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
                                                      
            torch.save(tokenized_datasets, saved_tokenized_datasets_fp)
            logger.info(f"✅ saved tokenized_data to {saved_tokenized_datasets_fp}")

        if "train" not in tokenized_datasets and training_args.do_train:
            raise ValueError("--do_train requires a train dataset")
        if "validation" not in tokenized_datasets and training_args.do_eval:
            raise ValueError("--do_eval requires a validation dataset")

        return tokenized_datasets


def get_lm_dataset(training_args, data_args, model_args, tokenizer):
    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)
    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        base_cache_dir = f"{model_args.cache_dir}/{data_args.dataset_name}/{data_args.dataset_config_name}"
        saved_lm_datasets_fp = pathlib.Path(f"{base_cache_dir}/lm_data_{data_args.max_train_samples}train_{data_args.max_eval_samples}eval_{len(tokenizer)}vocab.pt")

        if not data_args.overwrite_cache and saved_lm_datasets_fp.exists() and saved_lm_datasets_fp.is_file():
            lm_datasets = torch.load(str(saved_lm_datasets_fp))
            logger.info(f"✅ loaded lm_data from {saved_lm_datasets_fp}")
        else:
            tokenized_datasets = preprocess_data(training_args, data_args, model_args, tokenizer)
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
            torch.save(lm_datasets, saved_lm_datasets_fp)
            logger.info(f"✅ saved lm_data to {saved_lm_datasets_fp}")
    return lm_datasets

def modify_model(adapter_args, data_args, model_args, tokenizer, model):
    def get_adapter_config(adapter_args, model_args):
        # modify here for new parameter efficient techniques associated with adapter-hub
        if adapter_args.adapter_config == "prefix_tuning":
            if model_args.lang_adapt_strategies == "prefix_tuning_flat":
                assert adapter_args.flat is True
            else:
                assert adapter_args.flat is False
            
            if model_args == "prompt_tuning":
                assert model_args.adapter_placement == "0"

            if model_args.adapter_placement == "all":
                adapter_config = PrefixTuningConfig(
                    encoder_prefix = adapter_args.encoder_prefix,
                    cross_prefix = adapter_args.cross_prefix,
                    flat = adapter_args.flat,
                    prefix_length = adapter_args.prefix_length,
                    bottleneck_size = adapter_args.bottleneck_size,
                    non_linearity = adapter_args.non_linearity,
                    dropout = adapter_args.prefix_dropout
                )
            else:
                adapters2use = set([int(i) for i in model_args.adapter_placement.split(",")])
                adapter_config = PrefixTuningConfig(
                    leave_out = [i for i in range(0,24) if not i in adapters2use],
                    encoder_prefix = adapter_args.encoder_prefix,
                    cross_prefix = adapter_args.cross_prefix,
                    flat = adapter_args.flat,
                    prefix_length = adapter_args.prefix_length,
                    bottleneck_size = adapter_args.bottleneck_size,
                    non_linearity = adapter_args.non_linearity,
                    dropout = adapter_args.prefix_dropout
                )

        elif adapter_args.adapter_config == "lora":
            adapter_config = LoRAConfig(
                selfattn_lora = adapter_args.selfattn_lora,
                intermediate_lora = adapter_args.intermediate_lora,
                output_lora = adapter_args.output_lora, 
                r = adapter_args.r_lora,
                alpha = adapter_args.alpha_lora,
                dropout = adapter_args.dropout_lora,
                init_weights = adapter_args.init_weights_lora,
                composition_mode = adapter_args.composition_mode_lora,
                attn_matrices = adapter_args.attn_matrices_lora,
            )
        
        elif adapter_args.adapter_config == "ia3":
            adapter_config = IA3Config(
                selfattn_lora = adapter_args.selfattn_lora,
                intermediate_lora = adapter_args.intermediate_lora,
                output_lora = adapter_args.output_lora, 
                r = adapter_args.r_ia3,
                alpha = adapter_args.alpha_ia3,
                dropout = adapter_args.dropout_ia3,
                init_weights = adapter_args.init_weights_ia3,
                composition_mode = adapter_args.composition_mode_ia3,
                attn_matrices = adapter_args.attn_matrices_ia3,
            )
        
        elif adapter_args.adapter_config == "ia3+inv":
            ia3_adapter_config = IA3Config(
                selfattn_lora = adapter_args.selfattn_lora,
                intermediate_lora = adapter_args.intermediate_lora,
                output_lora = adapter_args.output_lora, 
                r = adapter_args.r_ia3,
                alpha = adapter_args.alpha_ia3,
                dropout = adapter_args.dropout_ia3,
                init_weights = adapter_args.init_weights_ia3,
                composition_mode = adapter_args.composition_mode_ia3,
                attn_matrices = adapter_args.attn_matrices_ia3,
            )
            inv_adapter_config = PfeifferInvConfig(
                mh_adapter=False,
                output_adapter=False,
                original_ln_before=False,
                original_ln_after=False,
                ln_before=False,
                ln_after=False,
                inv_adapter="nice",
                inv_adapter_reduction_factor=2,
            )
            adapter_config = ConfigUnion(
                ia3_adapter_config,
                inv_adapter_config
            )


        elif adapter_args.adapter_config == "inv":
            adapter_config = PfeifferInvConfig(
                mh_adapter=False,
                output_adapter=False,
                original_ln_before=False,
                original_ln_after=False,
                ln_before=False,
                ln_after=False,
                inv_adapter="nice",
                inv_adapter_reduction_factor=2,
            )
        
        #### commenting out because we are not using adaptable adapters
        # elif adapter_args.adapter_config == "aa":
        #     adapter_config = AAConfig(
        #         ln_before=True, # applying layernorm before the adapter bottleneck is necessary 
        #                         # for training stability w.r.t. Rational activation
        #                         # otherwise we will have loss = 0. and numerators/denominators = nan.
        #         reduction_factor=adapter_args.adapter_reduction_factor,
        #         non_linearity="rational_5_4",
        #     )
        
        elif adapter_args.adapter_config == "compacterpp":
            adapter_config = CompacterPlusPlusConfig()
        
        else:
            # TODO: confirm with Vassilina what goes into this condition
            if model_args.adapter_placement == "all":
                adapter_config = AdapterConfig.load(
                    adapter_args.adapter_config,
                    non_linearity=adapter_args.adapter_non_linearity,
                    reduction_factor=adapter_args.adapter_reduction_factor        
                )
            else:
                adapters2use = set([int(i) for i in model_args.adapter_placement.split(",")])
                adapter_config = AdapterConfig.load(
                    adapter_args.adapter_config,
                    non_linearity=adapter_args.adapter_non_linearity,
                    reduction_factor=adapter_args.adapter_reduction_factor,
                    leave_out = [i for i in range(0,24) if not i in adapters2use]
                )
        return adapter_config


    print(f"✅ Use Adaptation Strategy: {model_args.lang_adapt_strategies}")
    
    # Setup adapters
    if adapter_args.train_adapter:
        task_name = data_args.dataset_name or "clm"
        task_name += f"_{adapter_args.adapter_config}_{adapter_args.language}"
        # check if adapter already exists, otherwise add it
        if task_name not in model.config.adapters:
            adapter_config = get_adapter_config(adapter_args, model_args)
            # load a pre-trained from Hub if specified
            if adapter_args.load_adapter:
                model.load_adapter(
                    adapter_args.load_adapter,
                    config=adapter_config,
                    load_as=task_name,
                )
            else:
                model.add_adapter(task_name, config=adapter_config)
        
        # optionally load a pre-trained language adapter
        if adapter_args.load_lang_adapter:
            # resolve the language adapter config
            lang_adapter_config = AdapterConfig.load(
                adapter_args.lang_adapter_config,
                non_linearity=adapter_args.lang_adapter_non_linearity,
                reduction_factor=adapter_args.lang_adapter_reduction_factor,
            )
            # load the language adapter from Hub
            lang_adapter_name = model.load_adapter(
                adapter_args.load_lang_adapter,
                config=lang_adapter_config,
                load_as=adapter_args.language,
            )
        else:
            lang_adapter_name = None
        
        # Freeze all model weights except of those of this adapter
        model.train_adapter(task_name)
        
        # Set the adapters to be used in every forward pass
        #if lang_adapter_name:
        #    model.set_active_adapters(ac.Stack(lang_adapter_name, task_name))
        #else:
        #    model.set_active_adapters(task_name)

    else:
        if adapter_args.load_adapter or adapter_args.load_lang_adapter:
            raise ValueError(
                "Adapters can only be loaded in adapters training mode."
                "Use --train_adapter to enable adapter training"
            )

    print(f"✅ Use Embedding Strategy: {model_args.embedding_strategies}")

    if model_args.embedding_strategies == "overlap-replace" or model_args.embedding_strategies == "overlap-replace-breakdown":
        if not tokenizer.name_or_path == model_args.model_name_or_path:
            orig_tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
        else:
            raise Exception("Same tokenizer so overlap-replace doesn't make sense.")
        
        if hasattr(model.transformer, "wte"):
            # gpt-2
            ref_embedding = model.transformer.wte
        elif hasattr(model.transformer, "word_embeddings"):
            # bloom
            ref_embedding = model.transformer.word_embeddings
        else:
            raise Exception("Unsupported Model")

        model.resize_token_embeddings(len(tokenizer))
        overlap = set(tokenizer.vocab).intersection(set(orig_tokenizer.vocab))
        print(f"{len(overlap)} tokens overlapped")
        curr_vocab = tokenizer.vocab
        orig_vocab = orig_tokenizer.vocab
        for t in overlap:
            if hasattr(model.transformer, "wte"):
                model.transformer.wte.weight.data[curr_vocab[t]] = ref_embedding.weight[orig_vocab[t]]
            elif hasattr(model.transformer, "word_embeddings"):
                model.transformer.word_embeddings.weight.data[curr_vocab[t]] = ref_embedding.weight[orig_vocab[t]]
            else:
                raise Exception("Unsupported Model")
        
        # handling breakdown case
        if model_args.embedding_strategies == "overlap-replace-breakdown":
            not_overlap = set(tokenizer.vocab).difference(overlap)
            print(f"{len(not_overlap)} non-overlapped token to breakdown")
            for t in not_overlap:
                subtoken = orig_tokenizer.tokenize(t)
                merged_weight = sum([ref_embedding.weight[orig_vocab[st]] for st in subtoken]) / len(subtoken)

                if hasattr(model.transformer, "wte"):
                    model.transformer.wte.weight.data[curr_vocab[t]] = merged_weight
                elif hasattr(model.transformer, "word_embeddings"):
                    model.transformer.word_embeddings.weight.data[curr_vocab[t]] = merged_weight
                else:
                    raise Exception("Unsupported Model")

        model.tie_weights()

    elif model_args.embedding_strategies == "replace":
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()

    elif model_args.embedding_strategies == "extend":
        original_embedding_layer = model.get_input_embeddings()
        original_vocab_size = original_embedding_layer.weight.shape[0]
        print(f"Tokens for new languages: {len(tokenizer) - original_vocab_size}")
        model.resize_token_embeddings(len(tokenizer))
        model.tie_weights()
        
        embedding_layer = model.get_input_embeddings()
        # erases gradients for the original embedding layer, without using extra CUDA memory
        def zero_grad(grad):
            grad[:original_vocab_size, :] = 0
            return grad

        embedding_layer.weight.register_hook(lambda grad: zero_grad(grad))
    
    elif model_args.embedding_strategies == "original":
        pass # do nothing.

    if "reinit" in model_args.lang_adapt_strategies:
        print(f"❗️ Reinitialize model's weights")
        model.init_weights()

    trainable_params = 0
    frozen_params = 0
    emb_params = 0
    for name, param in model.named_parameters():
        if "word_embeddings" in name or "wte" in name or "wpe" in name or "lm_head" in name:
            if model_args.embedding_strategies == "original-frozen":
                param.requires_grad = False
            else:
                # model_args.embedding_strategies == "original"
                if model_args.lang_adapt_strategies == "sft" and ("lm_head" in name or "layernorm" in name):
                    # only freeze lm_head (since input and output embeddings are decoupled.)
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            emb_params += param.numel()
        
        elif model_args.lang_adapt_strategies is not None:
            if model_args.lang_adapt_strategies == "emb":
                param.requires_grad = False
            elif model_args.lang_adapt_strategies == "bitfit":
                if 'bias' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif model_args.lang_adapt_strategies == "bitfit+inv":
                if 'bias' in name or "inv" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            elif model_args.lang_adapt_strategies == "sft":
                if "layernorm" in name or "ln_f" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif "continual-pretrain" in model_args.lang_adapt_strategies:
                param.requires_grad = True

        if not param.requires_grad:
            print(f"🥶 Frozen layer '{name}'")
            frozen_params += param.numel()
        else:
            print(f"🚀 Trainable layer '{name}'")
            trainable_params += param.numel()

    print(f"Total trainable parameters: {trainable_params}")
    print(f"Total frozen parameters: {frozen_params}")
    print(f"Total emb parameters (wte, wpe): {emb_params}")
    print(f"❗️ Note: These figures are incorrect for masking-based methods such as sft and fish.")

def sft_get_maskable_params(model):
    # create masks for trainable layers 
    # TODO: asserting to make sure no unintended maskable layers for embedding layers   
    sft_maskable_params = [
        n for n, p in model.named_parameters()
        if "layernorm" not in n and p.requires_grad
    ]

    print("Maskable Params:")
    print(sft_maskable_params)

    return sft_maskable_params


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, ParamEfficientArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    
    training_args.data_dir = f'{training_args.output_dir}'

    # conditional checks
    assert model_args.lang_adapt_strategies in ("continual-pretrain", "continual-pretrain-reinit", 
                                                "emb", "emb-then-adpt", "pfeiffer", "pfeiffer+inv", 
                                                "compacterpp", "compacter", "lora", "aa",
                                                "ia3", "ia3+inv",
                                                "prefix_tuning", "prefix_tuning_flat", "prompt_tuning",
                                                "sft", "bitfit", "bitfit+inv", "fish")
    assert model_args.embedding_strategies in ("replace", "extend", "overlap-replace", "overlap-replace-breakdown", "original", "original-frozen")
    if adapter_args.train_adapter:
        assert model_args.lang_adapt_strategies not in ("emb", "continual-pretrain", "continual-pretrain-reinit", "bitfit"), f"remove --train_adapter for { model_args.lang_adapt_strategies} strategy."

    # setup trainer class
    trainer_class_mapping = {
        "continual-pretrain": Trainer,
        "continual-pretrain-reinit": Trainer,
        "emb": Trainer, 
        "emb-then-adpt": AdapterTrainer, 
        "pfeiffer": AdapterTrainer,
        "pfeiffer+inv": AdapterTrainer,
        "compacterpp": AdapterTrainer,
        "compacter": AdapterTrainer,
        "lora": AdapterTrainer,
        "ia3": AdapterTrainer,
        "ia3+inv": AdapterTrainer,
        "aa": AdapterTrainer,
        "prefix_tuning": AdapterTrainer,
        "prefix_tuning_flat": AdapterTrainer,
        "prompt_tuning": AdapterTrainer,
        "sft": LotteryTicketSparseFineTuner,
        "bitfit": Trainer,
        "bitfit+inv": Trainer,
        "fish": SparseUpdateTrainer,
    }

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"model_args {model_args}")
    logger.info(f"data_args {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Adapter parameters {adapter_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            pass
            #raise ValueError(
            #    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            #    "Use --overwrite_output_dir to overcome."
            #)
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = load_tokenizer(model_args)
    model = load_model(model_args, tokenizer)
    modify_model(adapter_args, data_args, model_args, tokenizer, model)
    
    # Preprocessing the datasets.
    lm_datasets = get_lm_dataset(training_args, data_args, model_args, tokenizer)
    if training_args.do_train:
        train_dataset = lm_datasets["train"]

    if training_args.do_eval:
        eval_dataset = lm_datasets["validation"]
    
    # masks for SFT and FISH
    if 'sft' in model_args.lang_adapt_strategies:
        sft_maskable_params = sft_get_maskable_params(model)
    else:
        sft_maskable_params = {}
    
    
    if 'fish' in model_args.lang_adapt_strategies:
        fish_sample_type, fish_grad_type = adapter_args.fish_mask_method.split("-")
        fish_mask = create_fish_mask_gradient(
            model,
            train_dataset,
            data_collator=default_data_collator,
            num_samples=adapter_args.fish_num_samples,
            keep_ratio=adapter_args.fish_keep_ratio,
            keep_num=adapter_args.fish_keep_num,
            sample_type=fish_sample_type,
            grad_type=fish_grad_type,
        )
    else:
        fish_mask = {}

    # Initialize our Trainer
    trainer_class = trainer_class_mapping[model_args.lang_adapt_strategies]

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        **({'sft_args': adapter_args} if 'sft' in model_args.lang_adapt_strategies else {}),
        **({'maskable_params': sft_maskable_params} if 'sft' in model_args.lang_adapt_strategies else {}),
        **({'mask': fish_mask} if 'fish' in model_args.lang_adapt_strategies else {}),
    )

    print("Model: 👇")
    print(model)
    
    # print("Embeddings at start of run:", model.get_input_embeddings().weight[250880:,:]) # get original weight for embedding layer
    # orig_embeddings = model.get_input_embeddings().weight.detach().clone() # clone original weight for embedding layer
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.add_callback(EarlyStoppingCallback(3))
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload # normally this part only saves the adapters? (TODO: check)

        # save embedding and positional embedding (which is not saved by trainer)
        
        # This part is used if we use initial BS 1b3 model  (the one used for experiments reported in the paper)
        if hasattr(trainer.model.transformer, "wte"):
            torch.save(trainer.model.transformer.wte, f'{trainer.args.output_dir}/embedding_wte.pt') # for sanity check
        if hasattr(trainer.model.transformer, "wpe"):
            torch.save(trainer.model.transformer.wpe, f'{trainer.args.output_dir}/embedding_wpe.pt')
        
        # this part is used for BLOOM models
        if hasattr(trainer.model.transformer, "word_embeddings"):
            torch.save(trainer.model.transformer.word_embeddings, f'{trainer.args.output_dir}/word_embeddings.pt')
            torch.save(trainer.model.transformer.word_embeddings_layernorm, f'{trainer.args.output_dir}/word_embeddings_layernorm.pt')

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        if 'sft' in model_args.lang_adapt_strategies:
            trainer.sft().save(f'{training_args.output_dir}/')
    
    
    # uncomment to test whether extending vocab gradient masking is working correctly. 
    # if model_args.embedding_strategies == "extend":
    #     print("Unsliced, post-training:", model.get_input_embeddings().weight) # get updated weight
    #     if not torch.equal(orig_embeddings[:250880, :], model.get_input_embeddings().weight[:250880, :]):
    #         raise ValueError("embedding layer is updated where it shouldn't....")

    #     if torch.equal(orig_embeddings[250880:, :], model.get_input_embeddings().weight[250880:, :]):
    #         print("original embeddings:", orig_embeddings[250880:, :])
    #         print("updated embeddings:", model.get_input_embeddings().weight[250880:, :])
    #         raise ValueError("embedding layer is not updated where it should....")


    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

#    if training_args.push_to_hub:
#        trainer.push_to_hub(**kwargs)
#    else:
#        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
