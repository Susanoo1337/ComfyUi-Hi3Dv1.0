"""Layer factories and attention dispatch shared by the Hi3D modules.

Linear layers come from an ``operations`` namespace, the way ComfyUI's own models
are built: inside ComfyUI that is ``comfy.ops.manual_cast`` (or the mixed-precision
ops for quantized checkpoints), so weights can be stored in fp8 or offloaded and
are cast to the activation dtype at call time. :class:`CastOps` gives the same
semantics without ComfyUI, which is what the tests run on.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn


def autocast_off(device_type: str):
    """Disable autocast on ``device_type``; a no-op where autocast does not exist."""
    if device_type in ("cuda", "cpu"):
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def autocast_fp16(device_type: str):
    """fp16 autocast on CUDA, as the release decoded; a no-op elsewhere."""
    if device_type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


class CastLinear(nn.Linear):
    """``nn.Linear`` that casts its parameters to the input dtype and device."""

    def reset_parameters(self) -> None:
        # Weights are always loaded from a checkpoint; skip the costly init.
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)
        return F.linear(x, weight, bias)


class CastOps:
    """Standalone stand-in for ``comfy.ops.manual_cast``."""

    Linear = CastLinear


class LayerNorm32(nn.Module):
    """Layer normalization computed in float32, returned in the input dtype.

    Parameter names match ``nn.LayerNorm`` so checkpoints load unchanged. The affine
    parameters stay float32 whatever dtype the rest of the model is stored in.
    """

    def __init__(
        self,
        channels: int,
        elementwise_affine: bool = True,
        eps: float = 1e-6,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.normalized_shape = (channels,)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(channels, device=device))
            self.bias = nn.Parameter(torch.zeros(channels, device=device))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = bias = None
        if self.weight is not None:
            weight = self.weight.to(device=x.device, dtype=torch.float32)
            bias = self.bias.to(device=x.device, dtype=torch.float32)
        with autocast_off(x.device.type):
            out = F.layer_norm(x.float(), self.normalized_shape, weight, bias, self.eps)
        return out.to(x.dtype)


def cast_param(param: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Return ``param`` on ``like``'s device and in its dtype."""
    return param.to(device=like.device, dtype=like.dtype)


_ATTENTION = None


def _resolve_attention():
    """Return ComfyUI's configured attention function, or ``None`` outside ComfyUI."""
    global _ATTENTION
    if _ATTENTION is None:
        try:
            from comfy.ldm.modules.attention import optimized_attention

            _ATTENTION = optimized_attention
        except Exception:  # pragma: no cover - depends on the host
            _ATTENTION = False
    return _ATTENTION or None


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Unmasked softmax attention with the default ``1/sqrt(D)`` scale.

    Args:
        q: Queries ``[B, H, Nq, D]``.
        k: Keys ``[B, H, Nk, D]``.
        v: Values ``[B, H, Nk, D]``.

    Returns:
        Attention output ``[B, Nq, H, D]``.
    """
    batch, heads, length, dim = q.shape
    backend = _resolve_attention()
    if backend is not None:
        # ComfyUI picks sage/flash/xformers/SDPA from its launch flags and splits
        # oversized batches itself; output is [B, N, H * D].
        out = backend(q, k, v, heads, skip_reshape=True)
        return out.reshape(batch, length, heads, dim)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2)


def empty_cache() -> None:
    """Release cached allocator blocks between memory-heavy decoder stages."""
    try:
        import comfy.model_management

        comfy.model_management.soft_empty_cache()
    except Exception:  # pragma: no cover - depends on the host
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def unique_rows(rows: torch.Tensor, return_counts: bool = False):
    """``torch.unique(rows, dim=0, return_inverse=True)`` for non-negative integer rows.

    Each row is packed into one int64 key (mixed radix over the per-column extents),
    so a 1-D unique replaces the row-wise one: the same lexicographic order and
    result, with a fraction of the memory and time on tens of millions of rows.

    Returns:
        ``(unique, inverse)`` or ``(unique, inverse, counts)``.
    """
    rows64 = rows.long()
    low = rows64.min(dim=0).values
    shifted = rows64 - low
    extents = (shifted.max(dim=0).values + 1).tolist()
    keys = shifted[:, 0].clone()
    for column in range(1, rows.shape[1]):
        keys.mul_(extents[column]).add_(shifted[:, column])
    result = torch.unique(keys, return_inverse=True, return_counts=return_counts)
    unique_keys = result[0]
    columns = []
    for column in reversed(range(rows.shape[1])):
        if column:
            columns.append(unique_keys % extents[column])
            unique_keys = unique_keys // extents[column]
        else:
            columns.append(unique_keys)
    unique = (torch.stack(columns[::-1], dim=1) + low).to(rows.dtype)
    return (unique, *result[1:])
