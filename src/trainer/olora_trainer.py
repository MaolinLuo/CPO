import os
import torch
import torch.nn as nn
from .sft_trainer import QwenSFTTrainer
from typing import Any, Union
from transformers.training_args import OptimizerNames
import safetensors.torch
from transformers.utils import (
    is_accelerate_available,
    is_sagemaker_mp_enabled,
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_mps_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    logger
)
if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
if is_accelerate_available():
    from accelerate.utils import DistributedType

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

class QwenSFTOLoRATrainer(QwenSFTTrainer):

    def __init__(self, *args, config_files=None, olora_lamda_1=0.05, olora_lamda_2=0.0,**kwargs):
        super(QwenSFTOLoRATrainer, self).__init__(*args, **kwargs)
        config_files = config_files.split(",") if config_files.strip() else []
        self.config_files = config_files or []
        self.past_num = len(self.config_files)
        self.olora_lamda_1 = olora_lamda_1
        self.olora_lamda_2 = olora_lamda_2

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()  
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available():
                torch.mps.empty_cache()
            elif is_torch_hpu_available():
                logger.warning(
                    "`torch_empty_cache_steps` is set but HPU device/backend does not support empty_cache()."
                )
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            olora_lamda_1 = self.olora_lamda_1
            olora_lamda_2 = self.olora_lamda_2
            past_num = self.past_num
            pre_loaded_weights = {}
            if past_num > 0:
                # Preload weights.
                for i in range(past_num):
                    config_path = self.config_files[i]
                    safe_file = os.path.join(config_path, "adapter_model.safetensors")
                    if self.model.device.type == "cuda":
                        device_idx = self.model.device.index  
                    else:
                        device_idx = self.model.device.type
                    weights = safetensors.torch.load_file(safe_file, device=device_idx)
                    pre_loaded_weights[i] = weights
            
                for name, param in self.model.named_parameters():
                    if "lora_A" in name:
                        for i in range(past_num):
                            weights = pre_loaded_weights[i]
                            for name_, param_ in weights.items():
                                if "lora_A" in name_ and name.split("lora_A")[0] == name_.split("lora_A")[0]:
                                    loss += torch.abs(torch.mm(param, param_.T)).sum()*olora_lamda_1 # [r * dim] * [dim * r]
                                    break # target modules have been matched
                    if olora_lamda_2 > 0.0 and "lora_" in name:
                        loss+= torch.norm(param, p=2) * olora_lamda_2

                del pre_loaded_weights
                torch.cuda.empty_cache()

            # Finally we need to normalize the loss for reporting if GA loss bug is not fixed during compute loss
            if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
                loss = loss / self.args.gradient_accumulation_steps
            
            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            return loss.detach()
