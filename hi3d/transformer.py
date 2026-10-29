"""Sparse latent flow transformer for image-conditioned mesh refinement."""

# Adapted from Microsoft TRELLIS under the MIT License.
# Modified for Hi3D: the released sparse inference architecture, rebuilt on
# pluggable casting layers (ComfyUI ``operations``), PyTorch attention and the
# PyTorch submanifold convolution in ``hi3d.sparse``.

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .ops import CastOps, LayerNorm32, attention, cast_param
from .sparse import SparseConv3d, SparseDownsample, SparseTensor, SparseUpsample

LATENT_MEAN = (
    -0.00155994, 0.03314649, -0.00307796, 0.00630751, -0.00835457, -0.03860179,
    0.02528263, 0.03740620, -0.06209833, -0.00539356, -0.02110271, 0.05126001,
    0.00753371, 0.02866821, 0.02772475, -0.02131735, -0.02799912, 0.01084617,
    0.03433723, -0.00903100, -0.00121899, -0.01944244, 0.01641873, -0.01063657,
    -0.01990618, 0.04856062, 0.01747232, -0.02387775, 0.03216169, -0.03732105,
    -0.03730354, -0.01948937,
)
"""Per-channel mean of the released structured latent."""

LATENT_STD = (
    0.61686146, 0.80115320, 0.85489200, 0.96384716, 0.77005910, 0.62859850,
    0.61541015, 0.74368310, 0.62762610, 1.01842280, 0.77113867, 0.62627990,
    0.68749195, 0.62698966, 0.75491240, 0.61577400, 0.63780660, 0.80406000,
    0.83846325, 0.69120390, 0.69227580, 0.79321617, 0.63475930, 0.63848300,
    0.67194235, 0.73519380, 0.84781533, 0.93012910, 0.62691440, 0.67906080,
    0.65198770, 0.77692040,
)
"""Per-channel standard deviation of the released structured latent."""


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture of :class:`Hi3DTransformer`; defaults are the released model."""

    in_channels: int = 32
    out_channels: int = 32
    model_channels: int = 2112
    cond_channels: int = 2048
    num_blocks: int = 24
    num_heads: int = 22
    mlp_ratio: float = 4.0
    num_io_res_blocks: int = 2
    io_block_channels: tuple[int, ...] = (256,)
    latent_mean: tuple[float, ...] = field(default=LATENT_MEAN)
    latent_std: tuple[float, ...] = field(default=LATENT_STD)

    def __post_init__(self) -> None:
        if self.model_channels % self.num_heads:
            raise ValueError("model_channels must be divisible by num_heads")
        head_dim = self.model_channels // self.num_heads
        if head_dim % 2 or head_dim < 6:
            raise ValueError("attention head dimension must be even and at least 6")
        if len(self.latent_mean) != self.in_channels or len(self.latent_std) != self.in_channels:
            raise ValueError("latent statistics must match in_channels")

    @property
    def patch_size(self) -> int:
        return 2 ** len(self.io_block_channels)


class TimestepEmbedder(nn.Module):
    """Sinusoidal scalar timestep embedding followed by a two-layer MLP."""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, *, ops, device=None):
        super().__init__()
        self.mlp = nn.Sequential(
            ops.Linear(frequency_embedding_size, hidden_size, bias=True, dtype=torch.float32, device=device),
            nn.SiLU(),
            ops.Linear(hidden_size, hidden_size, bias=True, dtype=torch.float32, device=device),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(timestep: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
        ).to(timestep.device)
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
        if dim % 2:
            embedding = torch.cat((embedding, torch.zeros_like(embedding[:, :1])), dim=-1)
        return embedding

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(timestep, self.frequency_embedding_size))


class RotaryPositionEmbedder(nn.Module):
    """Spatial rotary embedding with frequency scaling for long coordinate grids."""

    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2

    @staticmethod
    def _update_freqs(freqs: torch.Tensor, train_length: int, infer_length: int) -> torch.Tensor:
        wavelengths = freqs * train_length / (2 * math.pi)
        threshold = 24
        ratio = (wavelengths - 1) / (threshold - 1)
        ratio = ratio + (1 - ratio) * train_length / infer_length
        ratio = torch.where(wavelengths > threshold, 1.0, ratio)
        ratio = torch.where(wavelengths < 1, train_length / infer_length, ratio)
        return freqs * ratio

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        # Computed per call rather than kept as a buffer, so a model materialized
        # from the meta device never carries an uninitialized tensor.
        freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        freqs = 1.0 / (10000**freqs)
        max_length = int(indices.max().item())
        if max_length >= 64:
            freqs = self._update_freqs(freqs, train_length=64, infer_length=(64 + max_length) // 2)
        phases = torch.outer(indices, freqs.to(indices.device))
        return torch.polar(torch.ones_like(phases), phases)

    @staticmethod
    def _rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        rotated = x_complex * phases.unsqueeze(1)
        return torch.view_as_real(rotated).reshape(*rotated.shape[:-1], -1).to(x.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, indices: torch.Tensor):
        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)
        missing = self.hidden_size // 2 - phases.shape[-1]
        if missing > 0:
            padding = torch.polar(
                torch.ones(*phases.shape[:-1], missing, device=phases.device),
                torch.zeros(*phases.shape[:-1], missing, device=phases.device),
            )
            phases = torch.cat((phases, padding), dim=-1)
        return self._rotary_embedding(q, phases), self._rotary_embedding(k, phases)


class MultiHeadRMSNorm(nn.Module):
    """Per-head RMS normalization of a ``[N, H, D]`` query or key projection."""

    def __init__(self, dim: int, heads: int, device=None):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma.to(device=x.device, dtype=torch.float32)
        return (F.normalize(x.float(), dim=-1) * gamma * self.scale).to(x.dtype)


def _split_batches(x: SparseTensor):
    return [(span, x.feats[span]) for span in x.layout]


class SparseMultiHeadAttention(nn.Module):
    """Full self- or cross-attention over sparse queries."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: int | None = None,
        *,
        attention_type: str = "self",
        ops,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.channels = channels
        self.ctx_channels = ctx_channels or channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self._type = attention_type
        if attention_type == "self":
            self.to_qkv = ops.Linear(channels, channels * 3, bias=True, dtype=dtype, device=device)
            self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device)
            self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, num_heads, device=device)
            self.rope = RotaryPositionEmbedder(self.head_dim)
        else:
            self.to_q = ops.Linear(channels, channels, bias=True, dtype=dtype, device=device)
            self.to_kv = ops.Linear(self.ctx_channels, channels * 2, bias=True, dtype=dtype, device=device)
        self.to_out = ops.Linear(channels, channels, bias=True, dtype=dtype, device=device)

    def forward(self, x: SparseTensor, context: torch.Tensor | None = None) -> SparseTensor:
        heads, dim = self.num_heads, self.head_dim
        if self._type == "self":
            qkv = self.to_qkv(x.feats).reshape(-1, 3, heads, dim)
            q, k, v = qkv.unbind(dim=1)
            q, k = self.rope(q, k, x.coords[:, 1:])
            q = self.q_rms_norm(q)
            k = self.k_rms_norm(k)
            output = torch.empty_like(q)
            for span in x.layout:
                # One sequence per batch element: [1, H, N, D].
                qs, ks, vs = (t[span].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
                output[span] = attention(qs, ks, vs)[0]
        else:
            if context is None:
                raise ValueError("cross-attention requires context")
            query = self.to_q(x.feats).reshape(-1, heads, dim)
            kv = self.to_kv(context).reshape(context.shape[0], context.shape[1], 2, heads, dim)
            output = torch.empty_like(query)
            for batch, span in enumerate(x.layout):
                qs = query[span].transpose(0, 1).unsqueeze(0)
                ks = kv[batch, :, 0].transpose(0, 1).unsqueeze(0)
                vs = kv[batch, :, 1].transpose(0, 1).unsqueeze(0)
                output[span] = attention(qs, ks, vs)[0]
        return x.replace(self.to_out(output.reshape(-1, self.channels)))


class SparseFeedForwardNet(nn.Module):
    """Two-layer sparse MLP with tanh-approximated GELU."""

    def __init__(self, channels: int, mlp_ratio: float = 4.0, *, ops, dtype=None, device=None):
        super().__init__()
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            ops.Linear(channels, hidden, bias=True, dtype=dtype, device=device),
            nn.GELU(approximate="tanh"),
            ops.Linear(hidden, channels, bias=True, dtype=dtype, device=device),
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        return x.replace(self.mlp(x.feats))


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """Sparse self/cross-attention block with shared adaptive normalization."""

    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio=4.0, *, ops, dtype=None, device=None):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, device=device)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, device=device)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6, device=device)
        self.self_attn = SparseMultiHeadAttention(
            channels, num_heads, attention_type="self", ops=ops, dtype=dtype, device=device
        )
        self.cross_attn = SparseMultiHeadAttention(
            channels, num_heads, ctx_channels=ctx_channels, attention_type="cross",
            ops=ops, dtype=dtype, device=device,
        )
        self.mlp = SparseFeedForwardNet(channels, mlp_ratio, ops=ops, dtype=dtype, device=device)
        self.modulation = nn.Parameter(torch.zeros(1, 6 * channels, device=device))

    def forward(self, x: SparseTensor, mod: torch.Tensor, context: torch.Tensor) -> SparseTensor:
        modulation = self.modulation.to(device=mod.device, dtype=torch.float32)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            modulation + mod
        ).chunk(6, dim=1)
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.self_attn(h) * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        x = x + self.cross_attn(h, context)
        h = x.replace(self.norm3(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        return x + self.mlp(h) * gate_mlp


class SparseResBlock3d(nn.Module):
    """Timestep-modulated sparse residual block with optional paired resampling."""

    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: int | None = None,
        *,
        downsample: bool = False,
        upsample: bool = False,
        ops,
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.out_channels = out_channels or channels
        self.hidden_channels = self.out_channels // 2
        self.norm1 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6, device=device)
        self.norm2 = LayerNorm32(self.hidden_channels, elementwise_affine=False, eps=1e-6, device=device)
        self.conv1 = SparseConv3d(channels, self.hidden_channels, 3, dtype=dtype, device=device)
        self.conv2 = SparseConv3d(self.hidden_channels, self.hidden_channels, 3, dtype=dtype, device=device)
        self.linear = ops.Linear(self.hidden_channels, self.out_channels, bias=True, dtype=dtype, device=device)
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            ops.Linear(emb_channels, 2 * self.hidden_channels, bias=True, dtype=dtype, device=device),
        )
        self.skip_connection = (
            ops.Linear(channels, self.out_channels, bias=True, dtype=dtype, device=device)
            if channels != self.out_channels
            else None
        )
        self.updown = SparseDownsample(2) if downsample else SparseUpsample(2) if upsample else None

    def forward(self, x: SparseTensor, embedding: torch.Tensor) -> SparseTensor:
        scale, shift = self.emb_layers(embedding).to(x.dtype).chunk(2, dim=1)
        if self.updown is not None:
            x = self.updown(x)
        h = x.replace(F.silu(self.norm1(x.feats)))
        h = self.conv1(h)
        h = h.replace(self.norm2(h.feats)) * (1 + scale) + shift
        h = h.replace(F.silu(h.feats))
        h = self.conv2(h)
        h = h.replace(self.linear(F.silu(h.feats)))
        skip = x.feats if self.skip_connection is None else self.skip_connection(x.feats)
        return h + skip


class Hi3DTransformer(nn.Module):
    """Released sparse latent flow transformer.

    The torso (I/O resolution blocks and transformer blocks) holds its weights in
    ``dtype`` and computes in ``compute_dtype``; the timestep embedder, modulation
    head and the input/output projections stay float32, as in the reference.
    """

    def __init__(
        self,
        config: TransformerConfig,
        *,
        ops=CastOps,
        dtype: torch.dtype = torch.float16,
        compute_dtype: torch.dtype = torch.float16,
        device=None,
    ):
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        width = config.model_channels
        channels = list(config.io_block_channels)
        f32 = torch.float32

        self.t_embedder = TimestepEmbedder(width, ops=ops, device=device)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), ops.Linear(width, 6 * width, bias=True, dtype=f32, device=device)
        )
        self.input_layer = ops.Linear(config.in_channels, channels[0], bias=True, dtype=f32, device=device)

        torso = dict(ops=ops, dtype=dtype, device=device)
        self.input_blocks = nn.ModuleList()
        for current, following in zip(channels, channels[1:] + [width]):
            self.input_blocks.extend(
                SparseResBlock3d(current, width, **torso)
                for _ in range(config.num_io_res_blocks - 1)
            )
            self.input_blocks.append(
                SparseResBlock3d(current, width, out_channels=following, downsample=True, **torso)
            )

        self.blocks = nn.ModuleList(
            ModulatedSparseTransformerCrossBlock(
                width, config.cond_channels, config.num_heads, mlp_ratio=config.mlp_ratio, **torso
            )
            for _ in range(config.num_blocks)
        )

        self.out_blocks = nn.ModuleList()
        previous_channels = [width] + list(reversed(channels[1:]))
        for current, previous in zip(reversed(channels), previous_channels):
            self.out_blocks.append(
                SparseResBlock3d(previous * 2, width, out_channels=current, upsample=True, **torso)
            )
            self.out_blocks.extend(
                SparseResBlock3d(current * 2, width, out_channels=current, **torso)
                for _ in range(config.num_io_res_blocks - 1)
            )
        self.out_layer = ops.Linear(channels[0], config.out_channels, bias=True, dtype=f32, device=device)

    def forward(self, x: SparseTensor, timestep: torch.Tensor, cond: torch.Tensor) -> SparseTensor:
        """Predict latent flow for sparse ``x`` (float32) under image ``cond``."""
        if x.shape[0] != timestep.shape[0] or x.shape[0] != cond.shape[0]:
            raise ValueError("sparse input, timestep, and condition batches must match")
        dtype = self.compute_dtype
        h = x.replace(self.input_layer(x.feats)).type(dtype)
        timestep_embedding = self.t_embedder(timestep)
        convolution_embedding = timestep_embedding.to(dtype)
        skips = []
        for block in self.input_blocks:
            h = block(h, convolution_embedding)
            skips.append(h.feats)

        modulation = self.adaLN_modulation(timestep_embedding).to(dtype)
        context = cond.to(dtype)
        for block in self.blocks:
            h = block(h, modulation, context)

        for block, skip in zip(self.out_blocks, reversed(skips)):
            h = h.replace(torch.cat((h.feats, skip), dim=1))
            h = block(h, convolution_embedding)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        return x.replace(self.out_layer(h.feats.to(x.dtype)))

    def latent_stats(self, like: torch.Tensor):
        """Return the ``(mean, std)`` that map sampled latents back to VAE space."""
        mean = torch.tensor(self.config.latent_mean, dtype=like.dtype, device=like.device)
        std = torch.tensor(self.config.latent_std, dtype=like.dtype, device=like.device)
        return mean.reshape(1, -1), std.reshape(1, -1)


__all__ = ["Hi3DTransformer", "TransformerConfig", "LATENT_MEAN", "LATENT_STD", "cast_param"]
