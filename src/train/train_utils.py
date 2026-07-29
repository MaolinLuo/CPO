import transformers
import torch
import logging


def maybe_zero_3(param, ignore_status=False, name=None, device=torch.device('cpu')):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if type(device) is str:
        device = torch.device(device)
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach()
    else:
        param = param.detach()
    if device == param.device:
        return param.clone()
    else:
        return param.to(device)

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)


def count_parameters(model: "torch.nn.Module") -> tuple[int, int]:
    r"""Return the number of trainable parameters and number of all parameters in the model."""
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by itemsize
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "quant_storage") and hasattr(param.quant_storage, "itemsize"):
                num_bytes = param.quant_storage.itemsize
            elif hasattr(param, "element_size"):  # for older pytorch version
                num_bytes = param.element_size()
            else:
                num_bytes = 1

            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param

def freeze_lora_A_matrices(model, local_rank=-1):
    """
    Freeze all LoRA A matrices in the model.
    Compatible with DeepSpeed ZeRO.
    """
    from peft.tuners.lora import LoraLayer
    
    frozen_params = 0
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if hasattr(module, 'lora_A'):
                for adapter_name, lora_A in module.lora_A.items():
                    # For DeepSpeed, we need to access the actual parameter
                    if hasattr(lora_A, 'weight'):
                        lora_A.weight.requires_grad = False
                        frozen_params += lora_A.weight.numel()
    
    return frozen_params
                    
def initialize_lora_weights_custom(model, gradient_dict, lora_r, local_rank=-1):
    """
    Initialize LoRA weights using pre-computed SVD decomposition from gradients.
    Compatible with DeepSpeed ZeRO - handles parameter gathering/scattering.
    
    Args:
        model: The model with LoRA layers
        gradient_dict: Dict of pre-computed SVD results (A matrices) for initialization
        lora_r: LoRA rank (for verification purposes)
        local_rank: Local rank for distributed training
    """
    from peft.tuners.lora import LoraLayer
    import deepspeed
    
    # Check if using DeepSpeed
    is_deepspeed = hasattr(model, 'module')
    target_model = model.module if is_deepspeed else model
    
    initialized_count = 0
    for name, module in target_model.named_modules():
        if isinstance(module, LoraLayer):
            # Initialize A matrices with pre-computed SVD results
            if hasattr(module, 'lora_A'):
                for adapter_name, lora_A in module.lora_A.items():
                    if hasattr(lora_A, 'weight'):
                        param = lora_A.weight
                        
                        # Find matching SVD result for initialization
                        svd_weight = None
                        matched_grad_name = None
                        for grad_name, grad_svd in gradient_dict.items():
                            # grad_name like: "model.layers.0.self_attn.q_proj.weight"
                            # name like: "base_model.model.model.layers.0.self_attn.q_proj"
                            if grad_name.replace('.weight', '') in name:
                                svd_weight = grad_svd
                                matched_grad_name = grad_name
                                break
                        
                        if svd_weight is None:
                            continue  # Skip if no matching gradient found
                        
                        # Verify shape match
                        if svd_weight.shape != param.data.shape:
                            raise ValueError(f"Shape mismatch for {name}: SVD weight shape {svd_weight.shape} != param shape {param.data.shape}")
                        
                        # Handle DeepSpeed ZeRO: gather parameter if needed
                        if is_deepspeed and hasattr(param, 'ds_id'):
                            with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                                if local_rank in [0, -1]:
                                    with torch.no_grad():
                                        param.data.copy_(svd_weight.to(param.data.device))
                                        initialized_count += 1
                        else:
                            # Non-DeepSpeed or non-partitioned parameter
                            with torch.no_grad():
                                param.data.copy_(svd_weight.to(param.data.device))
                                initialized_count += 1
    
    if local_rank in [0, -1]:
        print(f"Successfully initialized {initialized_count} LoRA matrices with pre-computed SVD")
        if initialized_count == 0:
            raise ValueError("No LoRA matrices were initialized! Check gradient file and parameter names.")
    
    # Ensure all processes are synchronized
    if torch.distributed.is_initialized():
        torch.distributed.barrier()