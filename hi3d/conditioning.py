"""Image conditioning: subject framing, encoder inputs and token assembly.

The released conditioner concatenates L2-normalized DINOv2-large tokens of a
756x756 view with the L2-normalized pooled CLIP ViT-L/14 embedding of a 224x224
view, broadcast to every token: ``[1, 2917, 2048]``. Both encoders run in float32.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .image import prepare_image

DINO_SIZE = 756
CLIP_SIZE = 224
DINO_TOKENS = 1 + (DINO_SIZE // 14) ** 2
"""Class token plus a 54x54 patch grid."""

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def tensor_to_pil(image: torch.Tensor, mask: torch.Tensor | None = None) -> Image.Image:
    """Convert one ComfyUI image ``[H, W, C]`` in ``[0, 1]`` to a PIL image.

    Args:
        image: RGB or RGBA image.
        mask: Optional foreground mask ``[H, W]`` (1 = subject) used as alpha. It
            overrides an alpha channel already present in ``image``.
    """
    array = (image.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    if array.shape[-1] == 4 and mask is None:
        return Image.fromarray(array, mode="RGBA")
    rgb = Image.fromarray(array[..., :3], mode="RGB")
    if mask is None:
        return rgb
    mask = mask.detach().float().clamp(0, 1).cpu()
    if tuple(mask.shape) != tuple(array.shape[:2]):
        mask = F.interpolate(mask[None, None], size=array.shape[:2], mode="bilinear", align_corners=False)[0, 0]
    alpha = Image.fromarray((mask.numpy() * 255.0).round().astype(np.uint8), mode="L")
    rgba = rgb.convert("RGBA")
    rgba.putalpha(alpha)
    return rgba


def _pil_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).div_(255.0)


def _normalize(pixels: torch.Tensor, mean, std) -> torch.Tensor:
    mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (pixels - mean) / std


def encoder_inputs(image: Image.Image) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    """Frame ``image`` and build the normalized DINOv2 and CLIP pixel tensors.

    Returns:
        ``(prepared, dino_pixels, clip_pixels)``: the framed RGB image and float32
        tensors ``[1, 3, 756, 756]`` and ``[1, 3, 224, 224]`` on the CPU.
    """
    prepared = prepare_image(image)
    dino = F.interpolate(_pil_tensor(prepared), size=(DINO_SIZE, DINO_SIZE), mode="bilinear", align_corners=False)
    clip = _pil_tensor(prepared.convert("RGB").resize((CLIP_SIZE, CLIP_SIZE), resample=Image.Resampling.LANCZOS))
    return prepared, _normalize(dino, DINO_MEAN, DINO_STD), _normalize(clip, CLIP_MEAN, CLIP_STD)


@contextmanager
def release_position_interpolation(embeddings):
    """Interpolate DINOv2 position embeddings the way the release's encoder did.

    ``transformers`` 4.46, which the release was validated with, resizes the 37x37
    grid with ``size=(54, 54)``; the original DINOv2 code (and ComfyUI) passes a
    ``+0.1`` scale factor instead, which samples slightly different positions.
    This swaps in the former on a ComfyUI ``Dino2Embeddings`` for the duration.
    """

    def interpolate(x, h_pixels, w_pixels):
        pos = embeddings.position_embeddings.to(device=x.device, dtype=torch.float32)
        cls, patch = pos[:, :1], pos[:, 1:]
        side = int(round(patch.shape[1] ** 0.5))
        patch = patch.reshape(1, side, side, -1).permute(0, 3, 1, 2)
        size = (h_pixels // embeddings.patch_size, w_pixels // embeddings.patch_size)
        patch = F.interpolate(patch, size=size, mode="bicubic", align_corners=False)
        patch = patch.permute(0, 2, 3, 1).reshape(1, -1, patch.shape[1])
        return torch.cat((cls, patch), dim=1).to(x.dtype)

    embeddings.interpolate_pos_encoding = interpolate
    try:
        yield
    finally:
        del embeddings.interpolate_pos_encoding


def assemble_condition(dino_tokens: torch.Tensor, clip_pooled: torch.Tensor) -> torch.Tensor:
    """Concatenate normalized DINOv2 tokens with the broadcast CLIP embedding.

    Args:
        dino_tokens: Final-layernormed DINOv2 hidden states ``[1, 2917, 1024]``.
        clip_pooled: Post-layernormed CLIP class embedding ``[1, 1024]``, before the
            visual projection.
    """
    dino_tokens = F.normalize(dino_tokens.float(), p=2, dim=-1)
    clip_embedding = F.normalize(clip_pooled.float(), p=2, dim=-1)
    if dino_tokens.shape[1:] != (DINO_TOKENS, 1024):
        raise ValueError(
            f"DINOv2 tokens must be [1, {DINO_TOKENS}, 1024] (DINOv2-large, patch 14, no "
            f"registers), got {list(dino_tokens.shape)}"
        )
    if clip_embedding.shape[1:] != (1024,):
        raise ValueError(f"CLIP embedding must be [1, 1024] (ViT-L/14), got {list(clip_embedding.shape)}")
    clip_tokens = clip_embedding[:, None, :].expand(-1, dino_tokens.shape[1], -1)
    return torch.cat((dino_tokens, clip_tokens), dim=-1)
