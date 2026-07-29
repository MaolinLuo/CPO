import os
import torch
import torch.nn as nn
from .sft_trainer import QwenSFTTrainer
from typing import  Any, Union
from transformers.training_args import OptimizerNames
import safetensors.torch
from transformers.utils import (
    is_accelerate_available,
    is_sagemaker_mp_enabled,
)
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    PREFIX_CHECKPOINT_DIR,
    logger
)
from train.train_utils import  get_peft_state_non_lora_maybe_zero_3
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

class QwenSFTRegLoRATrainer(QwenSFTTrainer):

    def __init__(
        self, 
        *args, 
        config_files=None,               
        reg_lamda=25.0,        # RegLoRA regularization coefficient
        mask_ratio=0.02,       # Fraction of important parameters
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        config_files = config_files.split(",") if config_files and config_files.strip() else []
        self.config_files = config_files
        self.past_num = len(self.config_files)
        self.reg_lamda = reg_lamda
        self.mask_ratio = mask_ratio
        
        # Preload RegLoRA masks from all previous tasks.
        self.past_masks = self._load_past_task_masks()

    def _load_past_task_masks(self):
        """
        Load important-parameter masks from all previous tasks.
        Generate a missing mask from the corresponding LoRA weights.
        """
        past_masks = {}  # layer name -> accumulated mask matrix
        if self.past_num <= 0:
            return past_masks

        device = "cpu"
        logger.info(f"Loading RegLoRA masks or weights for {self.past_num} tasks")

        for task_idx in range(self.past_num):
            task_path = self.config_files[task_idx]
            adapter_file = os.path.join(task_path, "adapter_model.safetensors")
            mask_file = os.path.join(task_path, "reglora_masks.safetensors")

            # Load a saved mask when available.
            if os.path.exists(mask_file):
                masks = safetensors.torch.load_file(mask_file, device=device)
                for layer_name, mask in masks.items():
                    if layer_name not in past_masks:
                        past_masks[layer_name] = torch.zeros_like(mask)
                    past_masks[layer_name] += mask  # Accumulate masks from prior tasks.
                continue

            # Generate a missing mask from the LoRA update: delta_W = B @ A.
            if not os.path.exists(adapter_file):
                logger.warning(f"No weights found for task {task_idx}; skipping")
                continue
            
            lora_weights = safetensors.torch.load_file(adapter_file, device=device)
            task_masks = self._generate_masks_from_lora(lora_weights)
            
            # Save the generated mask for future runs.
            safetensors.torch.save_file(task_masks, mask_file)
            logger.info(f"Generated and saved the mask for task {task_idx}")
            
            for layer_name, mask in task_masks.items():
                if layer_name not in past_masks:
                    past_masks[layer_name] = torch.zeros_like(mask)
                past_masks[layer_name] += mask

        return past_masks

    def _generate_masks_from_lora(self, lora_weights):
        """
        Build an important-parameter mask from delta_W = B @ A.
        Mark the largest ``mask_ratio`` fraction by absolute magnitude.
        """
        masks = {}
        layer_names = set()
        
        # Extract LoRA layer names.
        for name in lora_weights.keys():
            if "lora_A" in name:
                layer_name = name.split(".lora_A")[0]
                layer_names.add(layer_name)

        # Compute the update and mask for each layer.
        for layer_name in layer_names:
            lora_A = lora_weights[f"{layer_name}.lora_A.default.weight"]
            lora_B = lora_weights[f"{layer_name}.lora_B.default.weight"]
            delta_W = torch.matmul(lora_B, lora_A)  # delta_W = B @ A

            # Select the largest ``mask_ratio`` fraction by absolute value.
            abs_delta = torch.abs(delta_W)
            num_elements = abs_delta.numel()
            top_k = max(1, int(num_elements * self.mask_ratio))
            
            # Flatten the update and determine the selection threshold.
            flat_abs = abs_delta.flatten()
            threshold = torch.topk(flat_abs, top_k)[0][-1]
            mask = (abs_delta >= threshold).float()

            masks[layer_name] = mask

        return masks

    def _save_checkpoint(self, model, trial):
        """Save the RegLoRA mask with the checkpoint."""
        super()._save_checkpoint(model, trial)
        if not self.args.lora_enable:
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Save the current task mask for the next task.
        if self.args.should_save and hasattr(self, "past_masks"):
            # Compute and save the current task mask.
            current_lora = {n: p for n, p in model.named_parameters() if "lora_" in n}
            current_masks = self._generate_masks_from_lora(current_lora)
            safetensors.torch.save_file(current_masks, os.path.join(output_dir, "reglora_masks.safetensors"))
            safetensors.torch.save_file(current_masks, os.path.join(run_dir, "reglora_masks.safetensors"))

        non_lora = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters(), require_grad_only=True)
        if self.args.should_save:
            torch.save(non_lora, os.path.join(output_dir, "non_lora_state_dict.bin"))
            self.model.base_model.config.to_json_file(os.path.join(output_dir, "config.json"))

    def training_step(
        self, model: nn.Module, inputs: dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        """
        Add the RegLoRA penalty to the language-model loss.
        L_total = L_lm + lambda * sum(abs(delta_W_j) * R_i)
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            base_loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # Compute the RegLoRA penalty.
        reg_loss = 0.0
        if self.past_num > 0 and self.past_masks:
            for name, param in model.named_parameters():
                if "lora_A" not in name:
                    continue
                layer_name = name.split(".lora_A")[0]
                if layer_name not in self.past_masks:
                    continue
                
                # Get the current layer's LoRA A and B matrices.
                lora_A = param
                lora_B = dict(model.named_parameters())[f"{layer_name}.lora_B.default.weight"]
                delta_W = torch.matmul(lora_B, lora_A)  # Current-task update.
                mask = self.past_masks[layer_name]      # Accumulated prior-task mask.

                mask = mask.to(delta_W.device)
                # Sum the masked absolute update element-wise.
                reg_loss += (torch.abs(delta_W) * mask).sum() / (mask.sum())

            # Apply the regularization coefficient.
            reg_loss = reg_loss * self.reg_lamda

        total_loss = base_loss + reg_loss

        if self.args.n_gpu > 1:
            total_loss = total_loss.mean()
        if not self.model_accepts_loss_kwargs and self.compute_loss_func is None:
            total_loss = total_loss / self.args.gradient_accumulation_steps

        kwargs = {}
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()
        if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
            kwargs["scale_wrt_gas"] = False

        self.accelerator.backward(total_loss, **kwargs)
        return total_loss.detach()
