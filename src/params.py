from dataclasses import dataclass, field
from typing import Optional

try:
    from accelerate.utils import ParallelismConfig as _PC
except Exception:
    class _PC:
        pass

import transformers.training_args as _ta
if not hasattr(_ta, "ParallelismConfig"):
    _ta.ParallelismConfig = _PC

from transformers import TrainingArguments as HFTrainingArguments
from trl import DPOConfig as DPOConfigTRL
from trl import GRPOConfig


@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="Qwen/Qwen2-VL-7B-Instruct")


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
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

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
    lora_target: str = field(default="all", metadata={"help": "List of namespan to target for LoRA"})
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1
    use_liger_kernel: bool = True
    #olora
    use_olora: bool = False
    olora_lamda_1: float =0.05
    olora_lamda_2: float =0.00
    config_files: str = field(default=None,metadata={"help":"The config file path of past LoRA checkpoint for olora/reglora"})
    #reglora
    use_reglora: bool = False
    reg_lamda: float = 25.0
    mask_ratio: float = 0.02
    #keeplora
    freeze_lora_A: bool = False
    init_lora_from_gradients: str = field(default=None)


@dataclass
class GRPOTrainingArguments:
    """VLM-specific training configuration for GRPO.
    
    This class contains parameters that are NOT part of TRL's GRPOConfig,
    specifically for handling vision-language models (freeze components, LoRA, quantization, etc.).
    
    Use together with GRPOConfig for complete GRPO training configuration.
    """
    # VLM-specific: freeze/unfreeze components
    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_merger: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    # Quantization settings
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

    # LoRA settings
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1

    # Component-specific learning rates
    vision_lr: Optional[float] = None
    merger_lr: Optional[float] = None

    # Continual learning mask regularization
    mask_path: Optional[str] = field(default=None, metadata={"help": "Path to importance mask .pt file for CL regularization."})
    mask_lambda: float = field(default=1e2, metadata={"help": "Regularization coefficient for mask loss."})

@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    prompt_path: str = field(
        default=None, metadata={"help": "Path to the prompt template yaml file."}
    )
    eval_path: str= field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    eval_image_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to the evaluation image data."}
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
    fps: Optional[int] = field(default=None, metadata={"help": "Frames per second for video data."})
    nframes: Optional[int] = field(default=None, metadata={"help": "Number of frames for video data."})