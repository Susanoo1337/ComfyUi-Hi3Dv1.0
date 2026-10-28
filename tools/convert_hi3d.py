#!/usr/bin/env python3
"""Repackage a downloaded Hi3D v1.0 model folder into single ComfyUI model files.

The release stores three ``model.safetensors`` files plus ``config.json`` files in
sub-folders. This script writes each one as a single named file with its config
embedded as ``hi3d_config`` metadata, optionally converting the transformers to a
smaller precision, so you can drop them into ComfyUI's model folders:

    python tools/convert_hi3d.py /path/to/Hi3Dv1.0 /path/to/ComfyUI/models --dtype fp8_scaled

writes ``diffusion_models/hi3d/hi3d_transformer_512_<dtype>.safetensors``,
``diffusion_models/hi3d/hi3d_transformer_1024_<dtype>.safetensors`` and
``vae/hi3d/hi3d_vae_fp32.safetensors``. It reads local files only.

Precisions for the transformers:
    fp16        the released precision (default)
    bf16        bfloat16 copy
    fp8_scaled  per-tensor scaled float8_e4m3fn transformer layers in ComfyUI's
                ``comfy_quant`` format; about half the size of fp16
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from safetensors.torch import load_file, save_file

_TORSO = ("input_blocks.", "blocks.", "out_blocks.")
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _layer_kind(sd: dict[str, torch.Tensor], key: str) -> str | None:
    """``"linear"``/``"conv"`` for weights and biases of torso layers, else ``None``."""
    if not key.startswith(_TORSO):
        return None
    stem, _, leaf = key.rpartition(".")
    weight = sd.get(stem + ".weight")
    if leaf not in ("weight", "bias") or weight is None:
        return None
    return {2: "linear", 5: "conv"}.get(weight.ndim)


def convert_transformer(sd: dict[str, torch.Tensor], dtype: str) -> dict[str, torch.Tensor]:
    """Convert torso layers to ``dtype``; norms, modulation and fp32 heads are kept."""
    half = torch.bfloat16 if dtype == "bf16" else torch.float16
    out = {}
    for key, tensor in sd.items():
        kind = _layer_kind(sd, key)
        if not tensor.is_floating_point() or kind is None:
            out[key] = tensor
        elif dtype == "fp8_scaled" and kind == "linear" and key.endswith(".weight"):
            module = key[: -len(".weight")]
            weight = tensor.float()
            scale = (weight.abs().amax() / _FP8_MAX).clamp(min=1e-12)
            out[key] = (weight / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
            out[module + ".weight_scale"] = scale.reshape(())
            config = json.dumps({"format": "float8_e4m3fn"}).encode("utf-8")
            out[module + ".comfy_quant"] = torch.tensor(list(config), dtype=torch.uint8)
        else:
            out[key] = tensor.to(half)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="Downloaded Hi3D v1.0 folder (with transformer_512/, transformer_1024/, vae/).")
    parser.add_argument("models_dir", help="ComfyUI models folder to write into, e.g. ComfyUI/models.")
    parser.add_argument("--dtype", choices=("fp16", "bf16", "fp8_scaled"), default="fp16")
    parser.add_argument("--skip-vae", action="store_true", help="Only convert the transformers.")
    args = parser.parse_args()

    jobs = [
        ("transformer_512", "diffusion_models", f"hi3d_transformer_512_{args.dtype}.safetensors"),
        ("transformer_1024", "diffusion_models", f"hi3d_transformer_1024_{args.dtype}.safetensors"),
    ]
    if not args.skip_vae:
        jobs.append(("vae", "vae", "hi3d_vae_fp32.safetensors"))

    for component, folder, filename in jobs:
        weights = os.path.join(args.source, component, "model.safetensors")
        config_path = os.path.join(args.source, component, "config.json")
        if not os.path.isfile(weights):
            print(f"missing {weights}", file=sys.stderr)
            return 1
        sd = load_file(weights)
        if component == "vae":
            sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
        else:
            sd = convert_transformer(sd, args.dtype)
        metadata = {"hi3d_component": component}
        if os.path.isfile(config_path):
            metadata["hi3d_config"] = json.dumps(_read_json(config_path))
        target_dir = os.path.join(args.models_dir, folder, "hi3d")
        os.makedirs(target_dir, exist_ok=True)
        target = os.path.join(target_dir, filename)
        save_file(sd, target, metadata=metadata)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
