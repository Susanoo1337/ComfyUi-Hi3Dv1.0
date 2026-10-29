"""Sparse PTv3 mesh decoder (``SparconvVAE``) and the ``Point`` container it uses.

Module and parameter names are load-bearing: they define the checkpoint key layout.
"""

# Adapted from Pointcept / Point Transformer V3 (Xiaoyang Wu et al.) under the
# MIT License. Modified for SPARC3D: specialized for the sparse mesh
# autoencoder, with the training paths, drop-path, octree branch and ``addict``
# dependency removed. Modified for Hi3D: spconv and flash-attn are replaced by
# the PyTorch submanifold convolution in ``hi3d.sparse`` and PyTorch attention,
# and linear layers come from pluggable casting ``operations``.

from __future__ import annotations

import functools
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from .ops import CastOps, LayerNorm32, attention, autocast_off, empty_cache
from .serialization import encode
from .sparse import SubMConv3d, neighbor_map

_RESNET_SKIP_POINTS = 12_000_000
"""Token count above which a residual block passes its input through and warns.

The released weights were measured with this guard active, so it is reproduced
behavior rather than a tuning knob."""

_GROUP_NORM_SPLIT = 8
"""Groups normalized per iteration; bounds temporaries without changing the result."""

WINDOW_ATTENTION_DTYPE: torch.dtype | None = None
"""Precision of the decoder's window attention; ``None`` means fp16 on CUDA (as the
release's flash attention ran) and the activation dtype elsewhere."""


# ---------------------------------------------------------------------------
# Batch/offset helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def offset2bincount(offset: Tensor) -> Tensor:
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.no_grad()
def offset2batch(offset: Tensor) -> Tensor:
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.no_grad()
def batch2offset(batch: Tensor) -> Tensor:
    return torch.cumsum(batch.bincount(), dim=0).long()


def deep_clone(value: Any) -> Any:
    """Copy a nested container, sharing tensors rather than duplicating them."""
    if isinstance(value, dict):
        return {key: deep_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [deep_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(deep_clone(item) for item in value)
    if isinstance(value, Tensor):
        return value
    return deepcopy(value)


def force_fp32(mode: str = "convert"):
    """Run a ``Point``-taking forward in float32 with autocast disabled.

    ``enforce`` raises unless the input is already float32, ``convert`` leaves the
    result in float32 and ``convert_back`` restores the input dtype on the result.
    """
    if mode not in {"enforce", "convert", "convert_back"}:
        raise ValueError(f"Invalid mode: {mode}")

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, point, *args, **kwargs):
            original_dtype = None
            if mode == "enforce":
                if point.dtype != torch.float32:
                    raise TypeError(
                        f"{type(self).__name__} requires float32 input, got {point.dtype}"
                    )
            elif point.dtype != torch.float32:
                original_dtype = point.dtype
                point = point.float()
            device_type = point.device.type if isinstance(point, Tensor) else point.feat.device.type
            with autocast_off(device_type):
                result = func(self, point, *args, **kwargs)
            if mode == "convert_back" and original_dtype is not None:
                result = result.to(original_dtype)
            return result

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Sparse convolution state
# ---------------------------------------------------------------------------


class ConvTensor:
    """Features plus the coordinates and neighbour cache that convolutions share."""

    def __init__(self, features: Tensor, indices: Tensor, cache: dict | None = None):
        self.features = features
        self.indices = indices
        self.cache = cache if cache is not None else {}

    def replace_feature(self, features: Tensor) -> "ConvTensor":
        if features.shape[0] != self.features.shape[0]:
            raise ValueError("replacement features must preserve the sparse row count")
        return ConvTensor(features, self.indices, self.cache)

    def neighbors(self, kernel_size: int) -> Tensor:
        key = f"subm{kernel_size}"
        if key not in self.cache:
            self.cache[key] = neighbor_map(self.indices, kernel_size)
        return self.cache[key]


class Conv3d(SubMConv3d):
    """Submanifold convolution over a :class:`ConvTensor`.

    Computes in the stored weight precision whatever the activation dtype, as the
    reference spconv override did, and returns the activation dtype.
    """

    def forward(self, input: ConvTensor) -> ConvTensor:
        feats = self.convolve(input.features, input.neighbors(self.kernel_size))
        return input.replace_feature(feats)


# ---------------------------------------------------------------------------
# Point container
# ---------------------------------------------------------------------------


class AttrDict(dict):
    """Dict whose keys are also readable and writable as attributes."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(key) from None


class Point(AttrDict):
    """Sparse voxel point cloud with a packed feature tensor and a declared layout."""

    ALL_LAYOUTS: Dict[str, Any] = {
        "sdf": {"shape": (8, 1), "size": 8, "denorm": lambda x: x, "act": lambda x: torch.tanh(x)},
        "deform": {"shape": (8, 3), "size": 8 * 3, "denorm": lambda x: x * 0.5, "act": lambda x: torch.tanh(x)},
    }

    def __init__(self, data_dict: Optional[dict] = None) -> None:
        super().__init__()
        if data_dict is not None:
            for key, value in data_dict.items():
                self[key] = AttrDict(value) if isinstance(value, dict) else value

    def initialization(self) -> None:
        if isinstance(self.resolution, Tensor):
            self.resolution = int(self.resolution[0])
        self["layouts"] = self._build_layouts()
        self["feat"] = self._build_feat()
        self["matrix"] = self._build_matrix()
        if "batch" not in self and "offset" in self:
            self["batch"] = offset2batch(self["offset"])
        elif "offset" not in self and "batch" in self:
            self["offset"] = batch2offset(self["batch"])
        self.is_serialized = False
        self.is_sparsified = False

    def serialization(self, order="z", depth: Optional[int] = None) -> None:
        if self.is_serialized:
            return
        if "batch" not in self or "grid_coord" not in self:
            raise KeyError("serialization requires 'batch' and 'grid_coord'")
        if depth is None:
            depth = int(self["grid_coord"].max().item()).bit_length()
        if depth * 3 + len(self["offset"]).bit_length() > 63 or depth > 16:
            raise ValueError("Serialization depth too high")
        if isinstance(order, str):
            order = [order]
        code = torch.stack([encode(self["grid_coord"], self["batch"], depth, order=o) for o in order])
        order_tensor = torch.argsort(code)
        inverse = torch.zeros_like(order_tensor).scatter_(
            dim=1,
            index=order_tensor,
            src=torch.arange(0, code.shape[1], device=order_tensor.device).repeat(code.shape[0], 1),
        )
        self["serialized_order"] = order_tensor
        self["serialized_inverse"] = inverse
        self.is_serialized = True

    def sparsify(self) -> None:
        if self.is_sparsified:
            return
        if not {"feat", "batch", "grid_coord"}.issubset(self.keys()):
            raise KeyError("sparsification requires 'feat', 'batch' and 'grid_coord'")
        indices = torch.cat([self["batch"].unsqueeze(-1).int(), self["grid_coord"].int()], dim=1)
        self["sparse_conv_feat"] = ConvTensor(self["feat"], indices.contiguous())
        self.is_sparsified = True

    def update_feat(self, feat: Tensor) -> "Point":
        self["feat"] = feat
        if self.is_sparsified:
            self["sparse_conv_feat"] = self["sparse_conv_feat"].replace_feature(feat)
        return self

    def float(self) -> "Point":
        return self.update_feat(self["feat"].float())

    def to(self, dtype: torch.dtype) -> "Point":
        if self.dtype != dtype:
            self.update_feat(self["feat"].to(dtype))
        return self

    def get_layout(self) -> Dict[str, Any]:
        return self["layouts"]

    def get_vert_feat(self, feat: Tensor):
        return self._get_attrs(feat, ["sdf", "deform"])

    def activate(self) -> None:
        layout = self.get_layout()
        activated = [entry["act"](self.get_attr_from_feat(self["feat"], layout, key)) for key, entry in layout.items()]
        self.update_feat(torch.cat(activated, dim=-1))

    @staticmethod
    def get_attr_from_feat(feat: Tensor, layouts: Dict[str, Any], key: str) -> Optional[Tensor]:
        if key not in layouts or layouts[key] is None:
            return None
        return feat[..., layouts[key]["range"][0]:layouts[key]["range"][1]]

    @staticmethod
    def grid_to_coord(grid: Tensor, matrix: Tensor) -> Tensor:
        grid = grid.float()
        stack = torch.cat([grid, grid.new_ones((grid.shape[0], 1))], dim=1)
        with autocast_off(grid.device.type):
            transformed = torch.matmul(matrix, stack.T).T
        return transformed[:, :3]

    def clone(self, **kwargs: Any) -> "Point":
        point = Point()
        for key, value in self.items():
            if key in kwargs or key == "sparse_conv_feat":
                continue
            point[key] = deep_clone(value)
        point.update(kwargs)
        point.initialization()
        return point

    def _build_layouts(self) -> Dict[str, Any]:
        if "layouts" in self:
            return self["layouts"]
        layouts: Dict[str, Any] = {}
        start = 0
        for key, entry in self.ALL_LAYOUTS.items():
            if key in self:
                layouts[key] = {**entry, "range": (start, start + entry["size"])}
                start += entry["size"]
        return layouts

    def _build_feat(self) -> Tensor:
        if "feat" in self:
            return self["feat"]
        raise KeyError("a Point needs 'feat' to decode")

    def _build_matrix(self) -> Tensor:
        if "matrix" in self:
            return self["matrix"]
        grid_coord = self["grid_coord"].float()
        center = grid_coord.mean(dim=0)
        centered = grid_coord - center
        scale = torch.norm(centered, dim=1).max() + 1e-8
        matrix = torch.eye(4, device=centered.device)
        matrix[:3, :3] /= scale
        matrix[:3, 3] = -center / scale
        return matrix

    def _get_attr(self, feat: Tensor, key: str):
        layouts = self.layouts
        if key not in layouts:
            return None, None
        attr = feat[:, layouts[key]["range"][0]:layouts[key]["range"][1]]
        attr = attr.reshape(-1, *layouts[key]["shape"])
        attr = layouts[key]["denorm"](attr)
        return attr, deepcopy(layouts[key])

    def _get_attrs(self, feat: Tensor, keys: List[str]):
        v_feat: List[Tensor] = []
        layouts: Dict[str, Any] = {}
        start = 0
        for key in keys:
            attr, layout = self._get_attr(feat, key)
            if attr is None:
                layouts[key] = None
                continue
            v_feat.append(attr)
            layout["range"] = (start, start + attr.shape[-1])
            layouts[key] = layout
            start += attr.shape[-1]
        return (torch.cat(v_feat, dim=-1) if v_feat else None), layouts

    @property
    def dtype(self) -> torch.dtype:
        return self["feat"].dtype

    @property
    def device(self) -> torch.device:
        return self["feat"].device


class PointModule(nn.Module):
    """Marker base class for modules whose ``forward`` takes and returns a ``Point``."""


class PointSequential(PointModule):
    """Sequential container dispatching on each child's kind."""

    def __init__(self, *args: nn.Module):
        super().__init__()
        for idx, module in enumerate(args):
            self.add_module(str(idx), module)

    def add(self, module: nn.Module, name: Optional[str] = None) -> None:
        self.add_module(str(len(self._modules)) if name is None else name, module)

    def forward(self, input: Point) -> Point:
        for module in self._modules.values():
            if isinstance(module, PointModule):
                input = module(input)
            elif isinstance(module, Conv3d):
                input.sparse_conv_feat = module(input.sparse_conv_feat)
                input.feat = input.sparse_conv_feat.features
            else:
                input.update_feat(module(input.feat))
        return input


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class Swish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class PointGroupNorm(PointModule):
    """Group normalization over the points of each batch element."""

    def __init__(self, channels: int, group: int = 32, min_group_channels: int = 4, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.group = group
        self.channels = channels
        if min_group_channels * self.group > channels:
            self.group = channels // min_group_channels
        if self.group < 1 or channels % self.group != 0:
            raise ValueError(f"channels {channels} must be divisible by group {self.group}")
        self.channels_per_group = channels // self.group
        self.weights = nn.Parameter(torch.ones(1, channels))
        self.bias = nn.Parameter(torch.zeros(1, channels))

    @force_fp32(mode="convert_back")
    def forward(self, point: Point) -> Point:
        feat = point.feat
        batch = point.batch.detach().clone()
        n, c = feat.shape
        b = int(batch[-1] + 1)
        g, cp = self.group, self.channels_per_group
        feat_group = feat.view(n, g, cp)
        feat_norm_all = feat.new_zeros((n, g, cp))
        counts = torch.bincount(batch, minlength=b).float().view(b, 1) * cp
        for start in range(0, g, _GROUP_NORM_SPLIT):
            end = min(start + _GROUP_NORM_SPLIT, g)
            feat_slice = feat_group[:, start:end, :]
            sum_feat = feat_slice.sum(dim=2)
            sum_feat_sq = feat_slice.square().sum(dim=2)
            batch_idx = batch.view(-1, 1).expand(-1, end - start).long()
            sums = feat.new_zeros((b, end - start)).scatter_add(0, batch_idx, sum_feat)
            sums_sq = feat.new_zeros((b, end - start)).scatter_add(0, batch_idx, sum_feat_sq)
            mean = sums.div_(counts + self.eps)
            var = sums_sq.div_(counts + self.eps).sub_(mean.square())
            inv_std = torch.rsqrt(var + self.eps)
            mean_exp = mean.index_select(0, batch).unsqueeze(2)
            inv_std_exp = inv_std.index_select(0, batch).unsqueeze(2)
            feat_norm_all[:, start:end, :] = feat_slice.sub(mean_exp).mul_(inv_std_exp)
            del feat_slice, sum_feat, sum_feat_sq, sums, sums_sq, mean, var, inv_std, mean_exp, inv_std_exp
        feat_norm_all = feat_norm_all.view(n, c)
        feat_norm_all.mul_(self.weights.to(device=feat.device, dtype=torch.float32))
        feat_norm_all.add_(self.bias.to(device=feat.device, dtype=torch.float32))
        point.update_feat(feat_norm_all)
        return point


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, *, ops, dtype=None):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = ops.Linear(in_channels, hidden_channels, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = ops.Linear(hidden_channels, out_channels, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SparseClsHead(PointModule):
    """Per-voxel occupancy head used on upsampling."""

    def __init__(self, in_channels: int, out_channels: int, dtype=None):
        super().__init__()
        self.fc = PointSequential(
            PointGroupNorm(in_channels),
            Swish(),
            Conv3d(in_channels, out_channels, kernel_size=3, bias=True, dtype=dtype),
        )

    @force_fp32(mode="convert")
    def forward(self, point: Point) -> Point:
        head_point = Point(dict(feat=point.feat, sparse_conv_feat=point.sparse_conv_feat, batch=point.batch))
        head_point.is_sparsified = True
        return self.fc(head_point)


class ConvOut(PointModule):
    """Output projection that splits off the eight per-corner SDF sign logits."""

    def __init__(self, in_channels: int, out_channels: int, dtype=None):
        super().__init__()
        self.conv = PointSequential(
            PointGroupNorm(in_channels),
            Swish(),
            Conv3d(in_channels, out_channels, kernel_size=3, bias=True, dtype=dtype),
        )

    @force_fp32(mode="convert")
    def forward(self, point: Point) -> Point:
        point = self.conv(point)
        point["sdf_sign"] = point.feat[:, :8]
        point.update_feat(point.feat[:, 8:])
        return point


class SparseZToF(PointModule):
    """Project latent channels back to decoder features."""

    def __init__(self, in_channels: int, out_channels: int, dtype=None):
        super().__init__()
        self.conv = PointSequential(Conv3d(in_channels, out_channels, kernel_size=3, bias=True, dtype=dtype))

    @force_fp32(mode="enforce")
    def forward(self, point: Point) -> Point:
        return self.conv(point)


class SparseResnetBlock(PointModule):
    """Pre-norm residual block of two sparse 3x3x3 convolutions."""

    def __init__(self, in_channels: int, out_channels: int, skip_points: int = _RESNET_SKIP_POINTS, *, ops, dtype=None):
        super().__init__()
        self.skip_points = skip_points
        self.block = PointSequential(
            PointGroupNorm(in_channels),
            Swish(),
            Conv3d(in_channels, out_channels, kernel_size=3, bias=True, dtype=dtype),
            PointGroupNorm(out_channels),
            Swish(),
            Conv3d(out_channels, out_channels, kernel_size=3, bias=True, dtype=dtype),
        )
        self.res_conv = ops.Linear(in_channels, out_channels, dtype=dtype) if in_channels != out_channels else None

    def forward(self, point: Point) -> Point:
        if point.feat.shape[0] > self.skip_points:
            warnings.warn(
                f"SparseResnetBlock skipped: {point.feat.shape[0]} points exceeds the "
                f"skip_points budget of {self.skip_points}.",
                stacklevel=2,
            )
            return point
        shortcut = point.feat if self.res_conv is None else self.res_conv(point.feat)
        point.update_feat(shortcut + self.block(point).feat)
        return point


class SerializedAttention(PointModule):
    """Windowed self-attention over patches of a space-filling-curve ordering."""

    def __init__(self, channels: int, num_heads: int, patch_size: int, order_index: int = 0, *, ops, dtype=None):
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels {channels} must be divisible by num_heads {num_heads}")
        self.channels = channels
        self.num_heads = num_heads
        self.order_index = order_index
        self.patch_size = patch_size
        self.qkv = ops.Linear(channels, channels * 3, bias=True, dtype=dtype)
        self.proj = ops.Linear(channels, channels, dtype=dtype)

    @torch.no_grad()
    def get_padding_and_inverse(self, point: Point):
        """Return, and cache on ``point``, the patch padding maps and sequence bounds."""
        pad_key, unpad_key, cu_seqlens_key = "pad", "unpad", "cu_seqlens_key"
        if pad_key not in point.keys() or unpad_key not in point.keys() or cu_seqlens_key not in point.keys():
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(bincount + self.patch_size - 1, self.patch_size, rounding_mode="trunc")
                * self.patch_size
            )
            mask_pad = bincount > self.patch_size
            bincount_pad = ~mask_pad * bincount + mask_pad * bincount_pad
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                unpad[_offset[i]:_offset[i + 1]] += _offset_pad[i] - _offset[i]
                if bincount[i] != bincount_pad[i]:
                    pad[_offset_pad[i + 1] - self.patch_size + (bincount[i] % self.patch_size):_offset_pad[i + 1]] = pad[
                        _offset_pad[i + 1] - 2 * self.patch_size + (bincount[i] % self.patch_size):_offset_pad[i + 1] - self.patch_size
                    ]
                pad[_offset_pad[i]:_offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(_offset_pad[i], _offset_pad[i + 1], step=self.patch_size, dtype=torch.int32, device=offset.device)
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])
        return point[pad_key], point[unpad_key], point[cu_seqlens_key]

    def forward(self, point: Point) -> Point:
        heads, channels = self.num_heads, self.channels
        dim = channels // heads
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)
        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        qkv = self.qkv(point.feat)[order]
        attn_dtype = WINDOW_ATTENTION_DTYPE
        if attn_dtype is None:
            attn_dtype = torch.float16 if qkv.device.type == "cuda" else qkv.dtype
        packed = qkv.to(attn_dtype).reshape(-1, 3, heads, dim)
        out = torch.empty((packed.shape[0], heads, dim), dtype=attn_dtype, device=packed.device)

        bounds = cu_seqlens.long()
        lengths = bounds[1:] - bounds[:-1]
        for length in torch.unique(lengths).tolist():
            starts = bounds[:-1][lengths == length]
            rows = (starts[:, None] + torch.arange(length, device=starts.device)[None]).reshape(-1)
            window = packed[rows].reshape(-1, length, 3, heads, dim)
            q, k, v = (window[:, :, i].transpose(1, 2) for i in range(3))
            out[rows] = attention(q, k, v).reshape(-1, heads, dim)

        feat = out.reshape(-1, channels).to(qkv.dtype)[inverse]
        point.feat = self.proj(feat)
        return point


class SparseAttentionBlock(PointModule):
    """Serialized attention block with a sparse conditional position encoding and MLP."""

    def __init__(self, channels, num_heads, patch_size=48, mlp_ratio=4.0, order_index=0, *, ops, dtype=None):
        super().__init__()
        self.channels = channels
        self.cpe = PointSequential(
            Conv3d(channels, channels, kernel_size=3, bias=True, dtype=dtype),
            ops.Linear(channels, channels, dtype=dtype),
            LayerNorm32(channels, eps=1e-5),
        )
        self.norm1 = PointSequential(LayerNorm32(channels, eps=1e-5))
        self.attn = SerializedAttention(
            channels=channels, patch_size=patch_size, num_heads=num_heads, order_index=order_index, ops=ops, dtype=dtype
        )
        self.norm2 = PointSequential(LayerNorm32(channels, eps=1e-5))
        self.mlp = PointSequential(
            MLP(channels, int(channels * mlp_ratio), channels, ops=ops, dtype=dtype)
        )

    def forward(self, point: Point) -> Point:
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm1(point)
        point = self.attn(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        point = self.norm2(point)
        point = self.mlp(point)
        point.feat = shortcut + point.feat
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedUnpooling(PointModule):
    """Double the grid resolution, keeping the children the occupancy head predicts."""

    def __init__(self, channels: int, *, ops, dtype=None):
        super().__init__()
        self.cls_head = SparseClsHead(channels, 8, dtype=dtype)
        self.proj = ops.Linear(channels, channels * 8, bias=False, dtype=dtype)
        self.inv_grid_coord = nn.Parameter(
            torch.tensor(
                [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
            ),
            requires_grad=False,
        )

    def forward(self, point: Point) -> Point:
        feat = self.proj(point.feat)
        pred_octree = self.cls_head(point).feat
        m, c = feat.shape[0], feat.shape[1] // 8
        mask = pred_octree.flatten() > 0
        feat = feat.view(m * 8, c)[mask]
        children = self.inv_grid_coord.to(device=point.grid_coord.device, dtype=torch.long)
        grid_coord = point.grid_coord.unsqueeze(1).long() * 2 + children.unsqueeze(0)
        grid_coord = grid_coord.view(-1, 3)[mask]
        batch = point.batch.unsqueeze(1).expand(-1, 8).flatten()[mask]
        point = point.clone(
            feat=feat, grid_coord=grid_coord, resolution=point.resolution * 2, batch=batch, offset=batch2offset(batch)
        )
        point.sparsify()
        return point


class Upsample(PointModule):
    def __init__(self, channels: int, *, ops, dtype=None):
        super().__init__()
        self.upsample = SerializedUnpooling(channels, ops=ops, dtype=dtype)

    def forward(self, point: Point) -> Point:
        # The point count grows up to eightfold here; release the freed stage first.
        empty_cache()
        return self.upsample(point)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecoderConfig:
    """Architecture of :class:`PTv3MeshDecoder`; defaults are the released decoder."""

    out_channels: int = 32
    order: Tuple[str, ...] = ("z", "z-trans")
    depths: Tuple[int, ...] = (2, 2, 2, 2)
    channels: Tuple[int, ...] = (128, 256, 512, 512)
    mid_depth: int = 2
    mid_num_head: int = 32
    z_channels: int = 32
    patch_size: int = 48
    mlp_ratio: float = 4.0


class PTv3MeshDecoder(PointModule):
    """Decode a coarse latent point cloud into per-corner SDF and deformation."""

    def __init__(self, config: DecoderConfig = DecoderConfig(), *, ops=CastOps, dtype=None):
        super().__init__()
        depths, channels = config.depths, config.channels
        self.num_stages = len(depths)
        self.order = [config.order] if isinstance(config.order, str) else list(config.order)
        if len(channels) != self.num_stages:
            raise ValueError("channels and depths describe different stage counts")
        kw = dict(ops=ops, dtype=dtype)

        mid_ch = channels[-1]
        self.z_to_f = SparseZToF(config.z_channels, mid_ch, dtype=dtype)
        self.mid = PointSequential()
        self.mid.add(SparseResnetBlock(mid_ch, mid_ch, **kw), name="block_in")
        for i in range(config.mid_depth):
            self.mid.add(
                SparseAttentionBlock(
                    channels=mid_ch,
                    num_heads=config.mid_num_head,
                    patch_size=config.patch_size,
                    mlp_ratio=config.mlp_ratio,
                    order_index=i % len(self.order),
                    **kw,
                ),
                name=f"block{i}",
            )
        self.mid.add(SparseResnetBlock(mid_ch, mid_ch, **kw), name="block_out")

        self.dec = PointSequential()
        block_out = channels[0]
        for s in reversed(range(self.num_stages)):
            dec = PointSequential()
            block_in = channels[s]
            block_out = channels[s - 1] if s > 0 else channels[0]
            for i in range(depths[s]):
                dec.add(SparseResnetBlock(block_in, block_out, **kw), name=f"block{i}")
                block_in = block_out
            if s != 0:
                dec.add(Upsample(block_out, **kw), name="up")
            self.dec.add(module=dec, name=f"dec{s}")
        self.conv_out = ConvOut(block_out, 8 + config.out_channels, dtype=dtype)

    def forward(self, point: Point) -> Point:
        if not point.is_sparsified:
            point.sparsify()
        point = self.z_to_f(point)
        if not point.is_serialized:
            point.serialization(order=self.order)
        point = self.mid(point)
        point = self.dec(point)
        return self.conv_out(point)


class SparconvVAE(nn.Module):
    """Decoder half of the released sparse mesh autoencoder."""

    def __init__(self, config: DecoderConfig = DecoderConfig(), *, ops=CastOps, dtype=None):
        super().__init__()
        expected = sum(Point.ALL_LAYOUTS[key]["size"] for key in ("sdf", "deform"))
        if config.out_channels != expected:
            raise ValueError(f"decoder out_channels must be {expected}, got {config.out_channels}")
        self.config = config
        self.decoder = PTv3MeshDecoder(config, ops=ops, dtype=dtype)

    def decode(self, grid_coord: Tensor, feats: Tensor, resolution: int) -> Point:
        """Decode a batch-1 latent grid into activated per-corner attributes.

        Args:
            grid_coord: Integer latent coordinates ``[N, 3]``.
            feats: Float32 latents ``[N, z_channels]`` in VAE space.
            resolution: Structure resolution the coordinates live on.
        """
        point = Point(
            {
                "grid_coord": grid_coord.int(),
                "feat": feats.float(),
                "sdf": None,
                "deform": None,
                "resolution": resolution,
                "offset": torch.tensor([grid_coord.shape[0]], dtype=torch.int64, device=grid_coord.device),
            }
        )
        point.initialization()
        x_hat = self.decoder(point)
        x_hat.activate()
        return x_hat


__all__ = ["SparconvVAE", "DecoderConfig", "Point"]
