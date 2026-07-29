import math
from contextlib import nullcontext

import torch
import torch.distributed as dist
from trl import GRPOTrainer


class CLGRPOTrainer(GRPOTrainer):
    """GRPOTrainer extended with importance-mask regularization for continual learning."""

    def __init__(self, mask_path: str, mask_lambda: float = 1.0, **kwargs):
        """
        Args:
            mask_path:   Path to masks/task_{k-1}.pt (new format: {masks, ref_weights}).
            mask_lambda: Regularization coefficient lambda.
            **kwargs:    Forwarded verbatim to GRPOTrainer.__init__.
        """
        self.mask_lambda = mask_lambda

        # -----------------------------------------------------------------------
        # Load mask file.
        #
        # New format (compute_importance_mask.py >= current version):
        #   {"masks": dict[str, BoolTensor], "ref_weights": dict[str, FloatTensor]}
        #
        # Old format (plain masks dict) — backward-compat for ZeRO-2 only.
        # -----------------------------------------------------------------------
        raw_saved = torch.load(mask_path, map_location="cpu", weights_only=True)

        if isinstance(raw_saved, dict) and "masks" in raw_saved and "ref_weights" in raw_saved:
            self.importance_mask: dict[str, torch.BoolTensor] = raw_saved["masks"]
            self.ref_weights: dict[str, torch.Tensor] = {
                k: v.float() for k, v in raw_saved["ref_weights"].items()
            }
        else:
            # Old format: just the masks dict (no embedded ref_weights).
            self.importance_mask: dict[str, torch.BoolTensor] = raw_saved
            self.ref_weights: dict[str, torch.Tensor] = {}

        # Precompute flat nonzero indices once to avoid .nonzero() on every step.
        # Shape: (|R_n|,) int64 tensor of global flat positions where mask is True.
        self._mask_flat_idx: dict[str, torch.Tensor] = {
            name: mask.view(-1).nonzero(as_tuple=True)[0]
            for name, mask in self.importance_mask.items()
        }

        # Per-step gradient contributions computed analytically in _compute_loss and
        # consumed by the gradient hooks during the subsequent backward pass.
        #
        # ZeRO-2 format:  { name: 1-D FloatTensor of length |R_n| }
        #   → hook injects at all masked positions of the full gradient.
        #
        # ZeRO-3 format:  { name: {"global_idx": LongTensor, "grad": FloatTensor} }
        #   → hook injects only at the positions handled by this rank's partition.
        self._pending_mask_grads: dict = {}
        self._mask_hook_handles: list = []
        self._zero3_hooks_registered: bool = False

        raw_model = kwargs["model"]

        # Detect whether ZeRO-3 is already active (from_pretrained ran under
        # deepspeed.zero.Init(), so params already carry ds_tensor).
        first_masked_param = next(
            (p for n, p in raw_model.named_parameters() if n in self.importance_mask),
            None,
        )
        is_zero3 = first_masked_param is not None and hasattr(first_masked_param, "ds_tensor")

        # Populate ref_weights from param.data for old-format masks (ZeRO-2 only).
        if not self.ref_weights:
            if is_zero3:
                raise RuntimeError(
                    "[CLGRPOTrainer] The mask file is in the old format (no embedded "
                    "ref_weights), but DeepSpeed ZeRO-3 is active and param.data is "
                    "partitioned/empty on each rank.  Please re-run "
                    "compute_importance_mask.py to regenerate the mask file — the "
                    "updated script saves ref_weights alongside the masks."
                )
            for name, param in raw_model.named_parameters():
                if name in self.importance_mask:
                    flat_idx = self._mask_flat_idx[name]
                    self.ref_weights[name] = (
                        param.data.view(-1)[flat_idx].detach().cpu().float().clone()
                    )

        # Register hooks.
        # ZeRO-2: register NOW (before super().__init__()) to be first in the hook queue.
        # ZeRO-3: skip here; _try_register_zero3_hooks() handles it on first _compute_loss.
        if not is_zero3:
            for name, param in raw_model.named_parameters():
                if name in self.importance_mask and param.requires_grad:
                    handle = param.register_hook(self._make_mask_grad_hook(name))
                    self._mask_hook_handles.append(handle)

        super().__init__(**kwargs)

        n_tensors = len(self.importance_mask)
        n_pos = sum(idx.numel() for idx in self._mask_flat_idx.values())
        print(
            f"[CLGRPOTrainer] mask loaded: {n_tensors} tensors, "
            f"{n_pos:,} masked positions, mask_lambda={mask_lambda}"
        )

    # ------------------------------------------------------------------
    # Gradient hook
    # ------------------------------------------------------------------

    def _make_mask_grad_hook(self, param_name: str):
        """
        Returns a gradient hook for one parameter.

        The hook adds the pre-computed mask gradient contribution to the GRPO
        gradient so DeepSpeed sees only one combined gradient per parameter.

        _pending_mask_grads[param_name] can be:
          • A 1-D tensor (ZeRO-2): gradient for all |R_n| masked positions, to be
            scattered into the full gradient via the flat mask indices.
          • A dict (ZeRO-3): {"global_idx": LongTensor, "grad": FloatTensor}
            containing only this rank's local-partition contribution.  The hook
            injects it at the global positions of the (full, pre-reduce-scatter)
            gradient that DeepSpeed passes to the hook.
        """
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if param_name not in self._pending_mask_grads:
                return grad

            pending = self._pending_mask_grads[param_name]

            if isinstance(pending, dict):
                # ------ ZeRO-3 path: partition-aware injection ------
                global_idx = pending["global_idx"].to(grad.device)
                g = pending["grad"].to(grad.device, dtype=grad.dtype)
                if global_idx.numel() == 0:
                    return grad
                # grad is the full parameter gradient (before reduce-scatter).
                # Add our contribution only at this rank's partition positions.
                out = grad.clone().reshape(-1)
                out[global_idx] = out[global_idx] + g
                return out.reshape(grad.shape)

            else:
                # ------ ZeRO-2 path: full gradient injection ------
                flat_idx = self._mask_flat_idx[param_name].to(grad.device)
                g = pending.to(grad.device, dtype=grad.dtype)
                grad_add = torch.zeros(grad.numel(), dtype=grad.dtype, device=grad.device)
                grad_add[flat_idx] = g
                return grad + grad_add.view(grad.shape)

        return hook

    # ------------------------------------------------------------------
    # Core override
    # ------------------------------------------------------------------

    def _compute_loss(self, model, inputs):
        # ZeRO-3: register hooks on the partitioned parameter objects on first call.
        if not self._zero3_hooks_registered:
            self._try_register_zero3_hooks(model)

        self._pending_mask_grads.clear()

        # ---- GRPO loss (single computation-graph path through model params) ----
        loss = super()._compute_loss(model, inputs)

        mode = "train" if self.model.training else "eval"
        unwrapped = self.accelerator.unwrap_model(model)
        normalizer = (
            self.current_gradient_accumulation_steps if mode == "train" else 1.0
        )

        mask_loss_value: float = 0.0

        with torch.no_grad():
            for name, param in unwrapped.named_parameters():
                if name not in self.importance_mask or not param.requires_grad:
                    continue

                flat_idx = self._mask_flat_idx[name].to(param.device)  # global flat indices
                ref = self.ref_weights[name].to(param.device, dtype=torch.float32)
                n_masked = flat_idx.numel()  # |R_n| (global, across all ranks)

                if hasattr(param, "ds_tensor"):
                    # ----------------------------------------------------------
                    # ZeRO-3 (any variant, including offload_optimizer):
                    #
                    # param.data is the local partition (or empty).  We access
                    # param.ds_tensor directly — it always holds this rank's slice
                    # of the full parameter, is always on the compute device, and
                    # does NOT require GatheredParameters (which conflicts with
                    # DeepSpeed's async prefetch under offload configs).
                    #
                    # Partition boundaries follow DeepSpeed's convention:
                    #   partition_size = ceil(ds_numel / world_size)
                    #   this rank covers [rank*P, min((rank+1)*P, ds_numel))
                    # ----------------------------------------------------------
                    full_numel = param.ds_numel
                    world_size = dist.get_world_size()
                    rank = dist.get_rank()
                    partition_size = math.ceil(full_numel / world_size)
                    ds_start = rank * partition_size
                    ds_end = min(ds_start + partition_size, full_numel)

                    # Restrict to mask positions that fall in this rank's partition.
                    in_partition = (flat_idx >= ds_start) & (flat_idx < ds_end)
                    if not in_partition.any():
                        continue

                    global_idx = flat_idx[in_partition]          # global positions in [ds_start, ds_end)
                    local_idx = global_idx - ds_start             # positions within the partition tensor
                    ref_local = ref[in_partition]

                    # param.ds_tensor: the actual partition data on GPU.
                    local_data = param.ds_tensor.to(param.device).float().view(-1)
                    diff = local_data[local_idx] - ref_local

                    mask_loss_value += diff.abs().sum().item() / max(n_masked, 1)

                    scale = self.mask_lambda / max(n_masked, 1) / normalizer
                    self._pending_mask_grads[name] = {
                        "global_idx": global_idx.cpu(),
                        "grad": (scale * torch.sign(diff)).detach().cpu(),
                    }

                else:
                    # ----------------------------------------------------------
                    # ZeRO-2 / no DeepSpeed: param.data is the full parameter.
                    # ----------------------------------------------------------
                    diff = param.data.view(-1)[flat_idx].float() - ref

                    mask_loss_value += diff.abs().sum().item() / max(n_masked, 1)

                    scale = self.mask_lambda / max(n_masked, 1) / normalizer
                    self._pending_mask_grads[name] = (scale * torch.sign(diff)).detach()

        # For ZeRO-3, each rank only sees its partition, so mask_loss_value is a
        # partial sum.  nanmean() across ranks approximates the per-rank average;
        # the exact total would be nansum().  Either way the logged value is a
        # consistent relative measure of how much the masked params have drifted.
        mask_loss_logged = self.mask_lambda * mask_loss_value
        self._metrics[mode]["mask_loss"].append(
            self.accelerator.gather(
                torch.tensor(mask_loss_logged, device=loss.device)
            ).nanmean().item()
        )

        return loss

    # ------------------------------------------------------------------
    # ZeRO-3 deferred hook registration
    # ------------------------------------------------------------------

    def _try_register_zero3_hooks(self, model) -> None:
        """
        Register gradient hooks on ZeRO-3 partitioned parameter objects.

        Called on the first _compute_loss invocation, by which time the DeepSpeed
        engine has been fully initialised and the parameter objects have been
        replaced with partitioned tensors (carrying ds_tensor).

        For ZeRO-2, the parameter objects are unchanged from __init__, so this
        is a no-op (is_zero3 will be False).
        """
        unwrapped = self.accelerator.unwrap_model(model)
        first_masked = next(
            (p for n, p in unwrapped.named_parameters() if n in self.importance_mask),
            None,
        )
        is_zero3 = first_masked is not None and hasattr(first_masked, "ds_tensor")
        if is_zero3:
            for name, param in unwrapped.named_parameters():
                if name in self.importance_mask and param.requires_grad:
                    param.register_hook(self._make_mask_grad_hook(name))
        self._zero3_hooks_registered = True
