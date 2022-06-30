from dataclasses import dataclass, field
from typing import Optional
# TODO: add attribution to Composable SFT authors/paper/code

"""A modified version of the SftArguments class that drops unused args and changes default values for language adaptation."""

@dataclass
class SftArguments:
    "Arguments pertaining to sparse fine-tuning configuration."""

    train_sft: bool = field(
        default=False, metadata={"help": "Whether to train sparse fine-tuning."}
    ) # newly added
    
    lang_ft: Optional[str] = field(
        default=None, metadata={"help": "Path to saved language SparseFineTuning."}
    )
    task_ft: Optional[str] = field(
        default=None, metadata={"help": "Path to saved task SparseFineTuning."}
    )
    
    sparse_ft_method: Optional[str] = field(
        default='LotteryTicket',
        metadata={"help": 'Sparse fine-tuning method. Can be LotteryTicket or Random.'},
    )

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

    freeze_embeddings: bool = field(
        default=False,
        metadata={
            "help": "Whether to freeze embeddings."
        },
    )
    freeze_head: bool = field(
        default=False,
        metadata={"help": "Whether to freeze language modeling head."},
    )
    untie_embeddings: bool = field(
        default=False,
        metadata={"help": "Whether to untie input and output embeddings."},
    )
    freeze_decoder: bool = field(
        default=False,
        metadata={"help": "Whether to freeze only output embeddings."},
    )
    freeze_layer_norm: bool = field(
        default=True,
        metadata={"help": "Whether to freeze layer normalisation parameters."},
    ) # changed from False to True
    
    ft_params_proportion: Optional[float] = field(
        default=None,
        metadata={
            "help": "The proportion of model parameters for which to learn non-zero differences during fine-tuning."
        },
    )
    ft_params_num: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of model parameters for which to learn non-zero differences during fine-tuning."
        },
    )
    n_ft_iterations: Optional[int] = field(
        default=5,
        metadata={
            "help": "The number of parameter selection iterations during fine-tuning."
        },
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
    full_ft_max_steps_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum number of steps per parameter selection iteration during full fine-tuning."
        },
    )
    sparse_ft_max_steps_per_iteration: Optional[int] = field(
        default=None,
        metadata={
            "help": "Maximum of steps per parameter selection iteration during sparse fine-tuning."
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
