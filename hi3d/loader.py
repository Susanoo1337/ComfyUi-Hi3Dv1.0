"""Build Hi3D models from user-supplied checkpoint files.

Nothing here downloads anything. Architecture is read from tensor shapes, so the
released ``model.safetensors`` files load directly; values the weights cannot
reveal (attention heads of the VAE, window size, curve order) come from an
optional ``config.json`` beside the weights or from ``hi3d_config`` safetensors
metadata, and otherwise default to the released architecture.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import fields
from typing import Any, Mapping

import torch
from torch import nn

from .ops import CastOps
from .sparse import SubMConv3d
from .transformer import Hi3DTransformer, TransformerConfig
from .vae import DecoderConfig, SparconvVAE

TRANSFORMER_MARKER = "blocks.0.self_attn.q_rms_norm.gamma"
VAE_MARKER = "decoder.z_to_f.conv.0.weight"
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "model.", "transformer.", "vae.", "first_stage_model.")
_TORSO_PREFIXES = ("input_blocks.", "blocks.", "out_blocks.")
_FLOAT_STORAGE = (torch.float16, torch.bfloat16, torch.float32)


def read_sidecar_config(path: str | None) -> dict[str, Any]:
    """Read ``config.json`` next to ``path``, or ``<stem>.json``, when present."""
    if not path:
        return {}
    folder = os.path.dirname(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    for candidate in (os.path.join(folder, stem + ".json"), os.path.join(folder, "config.json")):
        if os.path.isfile(candidate):
            with open(candidate, "r", encoding="utf-8") as handle:
                values = json.load(handle)
            if isinstance(values, Mapping):
                return dict(values)
    return {}


def metadata_config(metadata: Mapping[str, str] | None) -> dict[str, Any]:
    if not metadata or "hi3d_config" not in metadata:
        return {}
    try:
        values = json.loads(metadata["hi3d_config"])
    except (TypeError, ValueError):
        return {}
    return dict(values) if isinstance(values, Mapping) else {}


def normalize_state_dict(sd: dict[str, torch.Tensor], marker: str) -> dict[str, torch.Tensor]:
    """Strip a wrapper prefix such as ``model.diffusion_model.`` if the keys carry one."""
    if marker in sd:
        return sd
    for prefix in _PREFIXES:
        if prefix + marker in sd:
            return {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    return sd


def _count(sd, pattern: str) -> int:
    regex = re.compile(pattern)
    indices = {int(m.group(1)) for k in sd for m in [regex.match(k)] if m}
    return max(indices) + 1 if indices else 0


def _logical_dim(sd, key: str, axis: int, default: int) -> int:
    """Width of a weight axis, unless the tensor is bit-packed by a quant format."""
    tensor = sd.get(key)
    if tensor is None or not tensor.dtype.is_floating_point:
        return default
    return int(tensor.shape[axis])


def infer_transformer_config(sd, extra: Mapping[str, Any] | None = None) -> TransformerConfig:
    if TRANSFORMER_MARKER not in sd:
        raise ValueError(
            "this file is not a Hi3D transformer (expected transformer_512 or "
            "transformer_1024 weights)"
        )
    width = int(sd["blocks.0.norm2.weight"].shape[0])
    heads = int(sd[TRANSFORMER_MARKER].shape[0])
    hidden = int(sd["blocks.0.mlp.mlp.0.bias"].shape[0])

    stage_channels: list[int] = []
    per_stage = 0
    count = 0
    for i in range(_count(sd, r"input_blocks\.(\d+)\.")):
        in_ch = int(sd[f"input_blocks.{i}.norm1.weight"].shape[0])
        out_ch = int(sd[f"input_blocks.{i}.linear.bias"].shape[0])
        count += 1
        if out_ch != in_ch:
            stage_channels.append(in_ch)
            per_stage = per_stage or count
            count = 0
    values: dict[str, Any] = dict(
        in_channels=_logical_dim(sd, "input_layer.weight", 1, 32),
        out_channels=int(sd["out_layer.bias"].shape[0]),
        model_channels=width,
        cond_channels=_logical_dim(sd, "blocks.0.cross_attn.to_kv.weight", 1, 2048),
        num_blocks=_count(sd, r"blocks\.(\d+)\."),
        num_heads=heads,
        mlp_ratio=hidden / width,
        num_io_res_blocks=per_stage or 2,
        io_block_channels=tuple(stage_channels) or (256,),
    )
    if extra:
        for key in ("latent_mean", "latent_std"):
            if key in extra:
                values[key] = tuple(float(v) for v in extra[key])
        if "num_heads" in extra:
            values["num_heads"] = int(extra["num_heads"])
    return TransformerConfig(**values)


def infer_decoder_config(sd, extra: Mapping[str, Any] | None = None) -> DecoderConfig:
    if VAE_MARKER not in sd:
        raise ValueError("this file is not a Hi3D VAE (expected the vae/model.safetensors weights)")
    num_stages = _count(sd, r"decoder\.dec\.dec(\d+)\.")
    depths = tuple(_count(sd, rf"decoder\.dec\.dec{s}\.block(\d+)\.") for s in range(num_stages))
    channels = tuple(int(sd[f"decoder.dec.dec{s}.block0.block.0.weights"].shape[-1]) for s in range(num_stages))
    mid_ch = int(sd["decoder.z_to_f.conv.0.bias"].shape[0])
    mid_depth = _count(sd, r"decoder\.mid\.block(\d+)\.")
    values: dict[str, Any] = dict(
        out_channels=int(sd["decoder.conv_out.conv.2.bias"].shape[0]) - 8,
        depths=depths,
        channels=channels,
        mid_depth=mid_depth,
        z_channels=int(sd[VAE_MARKER].shape[-1]),
    )
    if mid_depth:
        values["mlp_ratio"] = int(sd["decoder.mid.block0.mlp.0.fc1.bias"].shape[0]) / mid_ch
    if extra:
        # Accept the release's {"model": {"decoder_cfg": {...}}} config.json as well.
        decoder = dict(extra.get("model", extra).get("decoder_cfg", extra.get("model", extra)))
        for key in ("mid_num_head", "patch_size"):
            if key in decoder:
                values[key] = int(decoder[key])
        if "order" in decoder:
            order = decoder["order"]
            values["order"] = (order,) if isinstance(order, str) else tuple(order)
    return DecoderConfig(**values)


def _storage_dtypes(model: nn.Module, torso_prefixes, weight_dtype, bias_dtype, conv_dtype) -> dict[str, torch.dtype]:
    """Storage dtype for every parameter: torso linear weights get ``weight_dtype``."""
    targets = {}
    for name, module in model.named_modules():
        for pname, _ in module.named_parameters(recurse=False):
            key = f"{name}.{pname}" if name else pname
            in_torso = key.startswith(torso_prefixes)
            if isinstance(module, nn.Linear) and in_torso:
                targets[key] = weight_dtype if pname == "weight" else bias_dtype
            elif isinstance(module, SubMConv3d) and in_torso:
                targets[key] = conv_dtype
            else:
                targets[key] = torch.float32
    return targets


def _check_keys(model: nn.Module, sd: Mapping[str, torch.Tensor], what: str, quantized: bool) -> None:
    expected = set(model.state_dict().keys())
    missing = sorted(expected - set(sd))
    unexpected = sorted(k for k in set(sd) - expected if not quantized)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"{len(missing)} missing (e.g. {missing[:3]})")
        if unexpected:
            details.append(f"{len(unexpected)} unexpected (e.g. {unexpected[:3]})")
        raise ValueError(f"{what} checkpoint does not match the architecture: " + "; ".join(details))


def _materialize(model_factory, sd, targets, quantized: bool, device) -> nn.Module:
    if quantized:
        # Quantized layers consume their scale tensors in _load_from_state_dict.
        model = model_factory(device)
        model.load_state_dict(sd, strict=False)
        return model
    with torch.device("meta"):
        model = model_factory(None)
    converted = {}
    for key, value in sd.items():
        target = targets.get(key)
        if target is not None and value.dtype.is_floating_point and value.dtype != target:
            value = value.to(target)
        converted[key] = value.to(device) if device is not None else value
    model.load_state_dict(converted, strict=True, assign=True)
    leftovers = [name for name, t in [*model.named_parameters(), *model.named_buffers()] if t.is_meta]
    if leftovers:
        raise RuntimeError(f"tensors left uninitialized after loading: {leftovers[:5]}")
    return model


def is_quantized(sd: Mapping[str, torch.Tensor]) -> bool:
    return any(k.endswith(".comfy_quant") for k in sd)


def build_transformer(
    sd: dict[str, torch.Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
    sidecar: Mapping[str, Any] | None = None,
    ops=CastOps,
    weight_dtype: torch.dtype | None = None,
    compute_dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
) -> Hi3DTransformer:
    """Create a :class:`Hi3DTransformer` from a state dict.

    Args:
        sd: Checkpoint tensors (``transformer_512`` or ``transformer_1024``).
        metadata: Safetensors metadata, consulted for ``hi3d_config``.
        sidecar: Parsed ``config.json`` of the release, if available.
        ops: Linear layer namespace (``comfy.ops.manual_cast`` inside ComfyUI).
        weight_dtype: Storage dtype of the torso linear weights; ``None`` keeps the
            checkpoint's dtype (float32 checkpoints are stored in ``compute_dtype``).
        compute_dtype: Activation dtype of the torso.
        device: Device to place the weights on.
    """
    sd = normalize_state_dict(sd, TRANSFORMER_MARKER)
    extra = {**(sidecar or {}), **metadata_config(metadata)}
    config = infer_transformer_config(sd, extra)
    quantized = is_quantized(sd)

    stored = sd["blocks.0.mlp.mlp.0.weight"].dtype
    if weight_dtype is None:
        # Quantized layers bring their own storage; the rest uses compute_dtype.
        keep = stored.is_floating_point and stored not in (torch.float32, torch.float64)
        weight_dtype = stored if keep and not quantized else compute_dtype
    plain_dtype = weight_dtype if weight_dtype in _FLOAT_STORAGE else compute_dtype

    def factory(dev, layer_ops=ops):
        return Hi3DTransformer(config, ops=layer_ops, dtype=weight_dtype, compute_dtype=compute_dtype, device=dev)

    # The probe uses plain layers: ComfyUI's may defer creating weights until load.
    with torch.device("meta"):
        probe = factory(None, CastOps)
    targets = _storage_dtypes(probe, _TORSO_PREFIXES, weight_dtype, plain_dtype, plain_dtype)
    _check_keys(probe, sd, "Hi3D transformer", quantized)
    del probe
    model = _materialize(factory, sd, targets, quantized, device)
    return model.eval().requires_grad_(False)


def build_vae(
    sd: dict[str, torch.Tensor],
    *,
    metadata: Mapping[str, str] | None = None,
    sidecar: Mapping[str, Any] | None = None,
    ops=CastOps,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> SparconvVAE:
    """Create the sparse VAE decoder from a state dict, stored in ``dtype``."""
    sd = normalize_state_dict(sd, VAE_MARKER)
    extra = {**(sidecar or {}), **metadata_config(metadata)}
    config = infer_decoder_config(sd, extra)
    quantized = is_quantized(sd)

    def factory(dev, layer_ops=ops):
        model = SparconvVAE(config, ops=layer_ops, dtype=dtype)
        return model.to(dev) if dev is not None else model

    with torch.device("meta"):
        probe = factory(None, CastOps)
    targets = _storage_dtypes(probe, ("decoder.",), dtype, dtype, dtype)
    # Group/layer norm parameters stay float32 (nn.Linear/SubMConv3d only above).
    _check_keys(probe, sd, "Hi3D VAE", quantized)
    del probe
    model = _materialize(factory, sd, targets, quantized, device)
    return model.eval().requires_grad_(False)


def config_dict(config) -> dict[str, Any]:
    return {f.name: getattr(config, f.name) for f in fields(config)}

