"""Sparse voxel tensors, submanifold convolution and paired resampling in PyTorch.

Replaces the spconv backend of the reference release. Convolution weights keep
spconv's ``[out, kx, ky, kz, in]`` layout so the released checkpoints load as-is,
and kernel offset ``(a, b, c)`` reads the neighbour at ``(x + a - r, y + b - r,
z + c - r)``, which is spconv's (and dense ``torch.nn.Conv3d``'s) correlation
convention.
"""

# Adapted from Microsoft TRELLIS under the MIT License (sparse tensor and
# resampling). Modified for Hi3D: the spconv backend is replaced by a
# gather-GEMM submanifold convolution.

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from typing import Any

import torch
from torch import nn

from .ops import autocast_off

_GATHER_BYTES = 512 * 1024 * 1024
"""Budget for one im2col gather buffer; bounds the convolution's temporary memory."""


def neighbor_map(coords: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Index of every kernel neighbour of every voxel, or ``-1`` when unoccupied.

    Args:
        coords: Integer coordinates ``[N, 4]`` as ``(batch, x, y, z)``.
        kernel_size: Odd cubic kernel extent.

    Returns:
        Int32 ``[N, kernel_size**3]``; column ``(a * k + b) * k + c`` holds the row of
        the voxel at offset ``(a - r, b - r, c - r)``.
    """
    radius = kernel_size // 2
    coords = coords.long()
    batch = coords[:, 0]
    xyz = coords[:, 1:]
    low = xyz.min(dim=0).values - radius
    xyz = xyz - low
    extent = xyz.max(dim=0).values + radius + 1
    e0, e1, e2 = (int(value) for value in extent.tolist())

    def encode(b: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        return ((b * e0 + x) * e1 + y) * e2 + z

    keys = encode(batch, xyz[:, 0], xyz[:, 1], xyz[:, 2])
    sorted_keys, order = torch.sort(keys)
    order = order.to(torch.int32)
    last = sorted_keys.shape[0] - 1

    result = torch.empty(
        (coords.shape[0], kernel_size**3), dtype=torch.int32, device=coords.device
    )
    offsets = product(range(-radius, radius + 1), repeat=3)
    for column, (dx, dy, dz) in enumerate(offsets):
        query = encode(batch, xyz[:, 0] + dx, xyz[:, 1] + dy, xyz[:, 2] + dz)
        position = torch.searchsorted(sorted_keys, query).clamp_(max=last)
        found = sorted_keys[position] == query
        result[:, column] = torch.where(found, order[position], -1)
    return result


class SubMConv3d(nn.Module):
    """Submanifold 3D convolution: output voxels are exactly the input voxels."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.empty(
                out_channels,
                kernel_size,
                kernel_size,
                kernel_size,
                in_channels,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def convolve(
        self,
        feats: torch.Tensor,
        neighbors: torch.Tensor,
        compute_dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Convolve ``feats`` over a precomputed :func:`neighbor_map`.

        Args:
            feats: Features ``[N, in_channels]``.
            neighbors: Neighbour indices ``[N, kernel_size**3]``.
            compute_dtype: GEMM dtype; defaults to the stored weight dtype, or to
                the feature dtype when the weights are stored in fp8.

        Returns:
            Features ``[N, out_channels]`` in ``feats``' dtype.
        """
        if compute_dtype is None:
            compute_dtype = self.weight.dtype
            if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                compute_dtype = feats.dtype
        volume = self.kernel_size**3
        weight = self.weight.to(device=feats.device, dtype=compute_dtype)
        weight = weight.reshape(self.out_channels, volume * self.in_channels).t()
        bias = None
        if self.bias is not None:
            bias = self.bias.to(device=feats.device, dtype=compute_dtype)

        source = feats.to(compute_dtype)
        out = torch.empty(
            (feats.shape[0], self.out_channels), dtype=compute_dtype, device=feats.device
        )
        row_bytes = volume * self.in_channels * source.element_size()
        chunk = max(4096, _GATHER_BYTES // max(row_bytes, 1))
        with autocast_off(feats.device.type):
            for start in range(0, feats.shape[0], chunk):
                stop = min(start + chunk, feats.shape[0])
                index = neighbors[start:stop].long()
                valid = index >= 0
                gathered = source[index.clamp(min=0)]
                gathered = gathered.masked_fill_(~valid.unsqueeze(-1), 0)
                gathered = gathered.reshape(stop - start, volume * self.in_channels)
                if bias is None:
                    out[start:stop] = gathered @ weight
                else:
                    out[start:stop] = torch.addmm(bias, gathered, weight)
        return out.to(feats.dtype)


class SparseTensor:
    """Sparse features with batch-first coordinates and shared spatial caches."""

    def __init__(
        self,
        feats: torch.Tensor,
        coords: torch.Tensor,
        shape: torch.Size | None = None,
        layout: list[slice] | None = None,
        *,
        scale: tuple[int, ...] = (1, 1, 1),
        spatial_cache: dict[str, dict[str, Any]] | None = None,
    ):
        """
        Args:
            feats: Features ``[N, ...]``, one row per occupied coordinate.
            coords: Coordinates ``[N, 4]`` as ``(batch, x, y, z)``, sorted by batch.
            shape: Logical shape; derived from ``coords`` and ``feats`` when omitted.
            layout: Per-batch row slices; recomputed from ``coords`` when omitted.
            scale: Cumulative downsampling factor relative to the original grid. It
                also keys :attr:`_spatial_cache`, so it differs between levels.
            spatial_cache: Cache shared with tensors over the same geometry: the
                neighbour maps of the convolutions and the inverse mappings a
                downsample hands to its paired upsample.
        """
        if coords.ndim != 2 or coords.shape[1] != 4:
            raise ValueError(f"expected coords [N, 4], got {tuple(coords.shape)}")
        if feats.shape[0] != coords.shape[0]:
            raise ValueError("features and coordinates must have the same row count")
        if coords.shape[0] == 0:
            raise ValueError("a SparseTensor cannot be empty")
        self.feats = feats
        self.coords = coords.int().contiguous()
        if layout is None:
            batch_size = int(self.coords[:, 0].max().item()) + 1
            layout = self._calculate_layout(self.coords, batch_size)
        self.layout = layout
        self.shape = shape or torch.Size((len(layout), *feats.shape[1:]))
        self.scale = tuple(scale)
        self._spatial_cache = spatial_cache if spatial_cache is not None else {}

    @staticmethod
    def _calculate_layout(coords: torch.Tensor, batch_size: int) -> list[slice]:
        counts = torch.bincount(coords[:, 0].long(), minlength=batch_size)
        offsets = torch.cumsum(counts, dim=0)
        layout = [
            slice(int(offsets[i] - counts[i]), int(offsets[i])) for i in range(batch_size)
        ]
        for batch, span in enumerate(layout):
            if not torch.all(coords[span, 0] == batch):
                raise ValueError("coordinates for each batch must be contiguous")
        return layout

    @property
    def dtype(self) -> torch.dtype:
        return self.feats.dtype

    @property
    def device(self) -> torch.device:
        return self.feats.device

    def replace(self, feats: torch.Tensor) -> "SparseTensor":
        """Return a tensor sharing coordinates and caches with new features."""
        if feats.shape[0] != self.feats.shape[0]:
            raise ValueError("replacement features must preserve the sparse row count")
        return SparseTensor(
            feats,
            self.coords,
            shape=torch.Size((self.shape[0], *feats.shape[1:])),
            layout=self.layout,
            scale=self.scale,
            spatial_cache=self._spatial_cache,
        )

    def reshape(self, *shape: int) -> "SparseTensor":
        return self.replace(self.feats.reshape(self.feats.shape[0], *shape))

    def type(self, dtype: torch.dtype) -> "SparseTensor":
        return self.replace(self.feats.to(dtype=dtype))

    def register_spatial_cache(self, key: str, value: Any) -> None:
        self._spatial_cache.setdefault(str(self.scale), {})[key] = value

    def get_spatial_cache(self, key: str) -> Any | None:
        return self._spatial_cache.get(str(self.scale), {}).get(key)

    def neighbors(self, kernel_size: int) -> torch.Tensor:
        """Neighbour map of this tensor's coordinates, built once per scale."""
        key = f"subm{kernel_size}"
        cached = self.get_spatial_cache(key)
        if cached is None:
            cached = neighbor_map(self.coords, kernel_size)
            self.register_spatial_cache(key, cached)
        return cached

    def _elementwise(
        self,
        other: "torch.Tensor | SparseTensor | float",
        operation: Callable[[torch.Tensor, Any], torch.Tensor],
    ) -> "SparseTensor":
        if isinstance(other, SparseTensor):
            other = other.feats
        elif isinstance(other, torch.Tensor) and other.ndim >= 1:
            if other.shape[0] == self.shape[0] and other.shape != self.feats.shape:
                other = sparse_batch_broadcast(self, other)
        return self.replace(operation(self.feats, other))

    def __add__(self, other):
        return self._elementwise(other, torch.add)

    def __mul__(self, other):
        return self._elementwise(other, torch.mul)


def sparse_batch_broadcast(input: SparseTensor, other: torch.Tensor) -> torch.Tensor:
    """Broadcast a batch-first tensor over sparse rows, in the features' dtype."""
    output = torch.empty_like(input.feats)
    for batch, span in enumerate(input.layout):
        output[span] = other[batch]
    return output


class SparseConv3d(nn.Module):
    """Coordinate-preserving submanifold convolution over a :class:`SparseTensor`.

    The ``conv`` child mirrors the reference wrapper so keys read
    ``<name>.conv.weight``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.conv = SubMConv3d(
            in_channels, out_channels, kernel_size, bias=bias, dtype=dtype, device=device
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        # The activation dtype is the compute dtype, matching the reference, whose
        # fp16 torso convolutions ran in fp16.
        feats = self.conv.convolve(
            x.feats, x.neighbors(self.conv.kernel_size), compute_dtype=x.dtype
        )
        return x.replace(feats)


def _factor_tuple(factor: int | Sequence[int], dimensions: int) -> tuple[int, ...]:
    values = tuple(factor) if isinstance(factor, Sequence) else (factor,) * dimensions
    if len(values) != dimensions or any(value <= 0 for value in values):
        raise ValueError("factor must provide one positive value per spatial dimension")
    return values


class SparseDownsample(nn.Module):
    """Average features that collapse onto the same downsampled coordinate."""

    def __init__(self, factor: int | Sequence[int]):
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, Sequence) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        dimensions = input.coords.shape[-1] - 1
        factor = _factor_tuple(self.factor, dimensions)
        columns = list(input.coords.unbind(dim=-1))
        for axis, value in enumerate(factor):
            columns[axis + 1] = columns[axis + 1] // value

        extents = [int(columns[axis + 1].max().item()) + 1 for axis in range(dimensions)]
        offsets = torch.cumprod(torch.tensor(extents[::-1]), 0).tolist()[::-1] + [1]
        codes = sum(column.long() * offset for column, offset in zip(columns, offsets))
        unique_codes, inverse = codes.unique(return_inverse=True)
        # include_self defaults to True, so the zero initial value joins the mean
        # (sum / (n + 1)). That is the reference's behavior and the weights expect it.
        new_feats = torch.scatter_reduce(
            torch.zeros(
                unique_codes.shape[0],
                input.feats.shape[1],
                device=input.device,
                dtype=input.dtype,
            ),
            dim=0,
            index=inverse[:, None].expand(-1, input.feats.shape[1]),
            src=input.feats,
            reduce="mean",
        )
        new_coords = torch.stack(
            [unique_codes // offsets[0]]
            + [
                (unique_codes // offsets[axis + 1]) % extents[axis]
                for axis in range(dimensions)
            ],
            dim=-1,
        )
        output = SparseTensor(
            new_feats,
            new_coords,
            shape=input.shape,
            scale=tuple(scale * value for scale, value in zip(input.scale, factor)),
            spatial_cache=input._spatial_cache,
        )
        output.register_spatial_cache(f"upsample_{factor}_coords", input.coords)
        output.register_spatial_cache(f"upsample_{factor}_layout", input.layout)
        output.register_spatial_cache(f"upsample_{factor}_idx", inverse)
        return output


class SparseUpsample(nn.Module):
    """Restore coordinates from a paired downsample cache."""

    def __init__(self, factor: int | Sequence[int]):
        super().__init__()
        self.factor = tuple(factor) if isinstance(factor, Sequence) else factor

    def forward(self, input: SparseTensor) -> SparseTensor:
        dimensions = input.coords.shape[-1] - 1
        factor = _factor_tuple(self.factor, dimensions)
        if any(scale % value for scale, value in zip(input.scale, factor)):
            raise ValueError(
                f"cannot upsample by {factor} from scale {input.scale}; "
                "SparseUpsample must mirror its paired SparseDownsample"
            )
        coords = input.get_spatial_cache(f"upsample_{factor}_coords")
        layout = input.get_spatial_cache(f"upsample_{factor}_layout")
        inverse = input.get_spatial_cache(f"upsample_{factor}_idx")
        if coords is None or layout is None or inverse is None:
            raise ValueError(
                "upsample cache not found; SparseUpsample must follow its "
                "paired SparseDownsample"
            )
        return SparseTensor(
            input.feats[inverse],
            coords,
            shape=input.shape,
            layout=layout,
            scale=tuple(scale // value for scale, value in zip(input.scale, factor)),
            spatial_cache=input._spatial_cache,
        )
