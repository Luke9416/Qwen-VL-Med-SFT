from dataclasses import dataclass, field
from typing import Optional, List

from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig as GRPOConfigTRL


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")
    lora_model_id: Optional[str] = field(default=None)

@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512
    lora_target_modules: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    
    
    # 添加缺失的参数
    use_liger: bool = field(
        default=False,
        metadata={"help": "Whether to use Liger kernel optimizations"}
    )
    
    
    
    # 评估相关参数 - 这些在HFTrainingArguments中已定义，但为了明确性可以显式声明默认值
    include_for_metrics: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of keys in inputs to pass to compute_metrics. "
                   "Set to ['inputs'] to pass all inputs, or specific keys like ['input_ids', 'attention_mask']"
        }
    )
    
    # 如果需要 batch_eval_metrics
    batch_eval_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to use batch evaluation for metrics"}
    )
    
    label_names: Optional[List[str]] = field(
        default=None,  # None 会使用默认值 ["labels"]
        metadata={
            "help": "The list of keys in your dictionary of inputs that correspond to the labels. "
                   "Default is ['labels']. For multiple labels, use e.g., ['labels', 'labels_2']"
        }
    )
    evaluation_strategy: str = field(
        default="no",
        metadata={"help": "The evaluation strategy to adopt during training. Options: 'no', 'steps', 'epoch'"}
    )
    eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Run an evaluation every X steps."}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/CPU for evaluation."}
    )
    metric_for_best_model: Optional[str] = field(
        default=None,
        metadata={"help": "The metric to use to compare two models."}
    )
    greater_is_better: Optional[bool] = field(
        default=None,
        metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether to load the best model at the end of training."}
    )
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the total amount of checkpoints. Deletes the older checkpoints."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to adopt during training. Options: 'no', 'epoch', 'steps'"}
    )
    save_steps: int = field(
        default=500,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    logging_first_step: bool = field(
        default=False,
        metadata={"help": "Log the first global step"}
    )
    logging_dir: Optional[str] = field(
        default=None,
        metadata={"help": "TensorBoard log directory. Will default to output_dir/logs"}
    )
    report_to: Optional[str] = field(
        default="none",
        metadata={"help": "The list of integrations to report results to. Options: 'tensorboard', 'wandb', 'none'"}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    remove_unused_columns: bool = field(
        default=True,
        metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: Optional[str] = field(
        default=None,
        metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    image_min_pixels: Optional[int] = field(default=3136)
    image_max_pixels: Optional[int] = field(default=12845056)
    video_min_pixels: Optional[int] = field(default=100352)
    video_max_pixels: Optional[int] = field(default=602112)
    image_resized_width: int = field(default=None)
    image_resized_height: int = field(default=None)
    video_resized_width: int = field(default=None)
    video_resized_height: int = field(default=None)
    fps: float = 1.0


@dataclass
class DPOArguments(DPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)

    max_seq_length: int = field(
        default=32768, # This is the default value of the qwen2-vl model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger: bool = True
    beta: float = field(
        default=0.1,
        metadata={"help": "The beta value for DPO."}
    )
    precompute_ref_log_probs: bool = field(
        default=False,
        metadata={"help": "Whether to precompute the reference log probabilities."}
    )
    dpo_loss:str = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."}
    )

@dataclass
class GRPOArguments(GRPOConfigTRL):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.999)
    adam_epsilon: float = field(default=1e-8)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    beta: float = field(
        default=0.04,
        metadata={
            "help": "KL coefficient. If `0.0`, the reference model is not loaded, reducing memory usage and improving "
            "training speed, but may be numerically unstable for long training runs."
        },
    )
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    min_p: Optional[float] = None
    repetition_penalty: float = 1.0
    max_completion_length: int = 256
    max_prompt_length: int = 512