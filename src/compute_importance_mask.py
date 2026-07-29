import argparse
import gc
import os

import torch
from transformers import (
    AutoConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
)


def load_full_model(model_path: str):
    print(f"  Loading model from {model_path} ...")
    config = AutoConfig.from_pretrained(model_path)
    kwargs = dict(dtype=torch.bfloat16, device_map="cpu")
    if config.model_type == "qwen3_vl_moe":
        return Qwen3VLMoeForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif config.model_type == "qwen3_vl":
        return Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    elif config.model_type == "qwen2_5_vl":
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, **kwargs)
    else:
        return Qwen2VLForConditionalGeneration.from_pretrained(model_path, **kwargs)


def load_state_dict_cpu(model_path: str) -> dict:
    model = load_full_model(model_path)
    sd = {k: v.cpu() for k, v in model.state_dict().items()}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return sd


def compute_mask(
    base_model: str,
    task_id: int,
    datasets: list,
    top_percent: float,
    checkpoint_dir: str,
) -> None:
    cur_dataset = datasets[task_id - 1]
    cur_model_path = os.path.join(checkpoint_dir, "training", cur_dataset)

    if task_id == 1:
        prev_model_path = base_model
    else:
        prev_dataset = datasets[task_id - 2]
        prev_model_path = os.path.join(checkpoint_dir, "training", prev_dataset)

    print(f"[compute_mask] Task {task_id}: {prev_model_path}  ->  {cur_model_path}")
    print(f"[compute_mask] top_percent={top_percent}%")

    # Load state dicts one at a time to minimise peak memory
    print("[compute_mask] Loading current task weights ...")
    cur_sd = load_state_dict_cpu(cur_model_path)
    print("[compute_mask] Loading previous task weights ...")
    prev_sd = load_state_dict_cpu(prev_model_path)

    # Collect absolute differences for all float parameters
    float_keys = [
        k for k in cur_sd
        if k in prev_sd and torch.is_floating_point(cur_sd[k])
    ]

    print(
        f"[compute_mask] Computing per-parameter top-{top_percent}% masks "
        f"for {len(float_keys)} tensors (floor count, no global threshold) ..."
    )

    # Build per-parameter boolean masks for the current task.
    # Each tensor is handled independently: k = floor(numel * top_percent / 100).
    # int() in Python truncates toward zero (== floor for positive values), so
    # the selected count is always <= top_percent of the tensor's elements.
    new_mask: dict[str, torch.BoolTensor] = {}
    total_params: int = 0
    total_masked_new: int = 0

    for key in float_keys:
        diff = (cur_sd[key].float() - prev_sd[key].float()).abs()
        n = diff.numel()
        total_params += n

        k = int(n * top_percent / 100.0)  # floor — never exceeds top_percent
        if k == 0:
            # Tensor too small or almost no update; skip entirely
            continue

        flat_diff = diff.flatten()
        # topk with sorted=False is faster; we only need the indices
        top_vals, top_indices = torch.topk(flat_diff, k, largest=True, sorted=False)

        valid_mask = top_vals > 0.0
        valid_indices = top_indices[valid_mask]
        
        actual_k = valid_indices.numel()
        if actual_k == 0:
            continue

        m = torch.zeros(n, dtype=torch.bool)
        m[valid_indices] = True
        new_mask[key] = m.view(diff.shape)
        total_masked_new += actual_k

    # Free prev_sd (no longer needed after diffs are computed); keep cur_sd alive
    # because we need it below to extract reference weights at the masked positions.
    del prev_sd
    gc.collect()

    print(
        f"[compute_mask] New task mask: {total_masked_new:,} / {total_params:,} = "
        f"{100.0 * total_masked_new / max(total_params, 1):.4f}% positions selected"
    )

    # Load and union with the previous accumulated mask (if it exists)
    mask_dir = os.path.join(checkpoint_dir, "masks")
    os.makedirs(mask_dir, exist_ok=True)
    prev_mask_path = os.path.join(mask_dir, f"task_{task_id - 1}.pt")

    if task_id > 1 and os.path.isfile(prev_mask_path):
        print(f"[compute_mask] Loading previous mask from {prev_mask_path} ...")
        prev_saved = torch.load(prev_mask_path, map_location="cpu", weights_only=True)
        # Support both old format (plain masks dict) and new format ({masks, ref_weights})
        prev_mask = prev_saved["masks"] if isinstance(prev_saved, dict) and "masks" in prev_saved else prev_saved
        for key, prev_m in prev_mask.items():
            if key in new_mask:
                new_mask[key] = new_mask[key] | prev_m
            else:
                new_mask[key] = prev_m
        del prev_saved, prev_mask
        gc.collect()
    elif task_id > 1:
        print(f"[compute_mask] Warning: expected previous mask at {prev_mask_path} but file not found. Proceeding with new mask only.")

    # Report coverage
    total_masked = sum(m.sum().item() for m in new_mask.values())
    print(f"[compute_mask] Total masked positions after union: {total_masked:,}  ({100.0 * total_masked / total_params:.2f}% of all params)")

    # Extract reference weights from cur_sd at ALL important positions (after union).
    # These are saved alongside the masks so that CLGRPOTrainer can load them directly
    # without touching param.data — which is unavailable under DeepSpeed ZeRO-3 because
    # from_pretrained runs inside deepspeed.zero.Init() and immediately partitions every
    # parameter across ranks, leaving param.data as an empty tensor on each rank.
    print("[compute_mask] Extracting reference weights at masked positions ...")
    ref_weights: dict[str, torch.Tensor] = {}
    for key, m in new_mask.items():
        if key in cur_sd:
            ref_weights[key] = cur_sd[key].float().view(-1)[m.view(-1)].clone()

    del cur_sd
    gc.collect()

    # Save masks + ref_weights together in a single file.
    save_path = os.path.join(mask_dir, f"task_{task_id}.pt")
    torch.save({"masks": new_mask, "ref_weights": ref_weights}, save_path)
    print(f"[compute_mask] Mask saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute importance mask for CL regularization.")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base (pre-training) model.")
    parser.add_argument("--task_id", type=int, required=True, help="1-based task index.")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, help="Ordered list of dataset names.")
    parser.add_argument("--top_percent", type=float, default=2.0, help="Percentage of parameters to mark as important.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/Qwen3-VL-2B/GRPO-CL",
        help="Root checkpoint directory (contains 'training/' and 'masks/' subdirs).",
    )
    args = parser.parse_args()

    assert 1 <= args.task_id <= len(args.datasets), (
        f"task_id {args.task_id} out of range for {len(args.datasets)} datasets."
    )

    compute_mask(
        base_model=args.base_model,
        task_id=args.task_id,
        datasets=args.datasets,
        top_percent=args.top_percent,
        checkpoint_dir=args.checkpoint_dir,
    )


if __name__ == "__main__":
    main()
