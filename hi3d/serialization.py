"""Z-order (Morton) serialization of integer grid coordinates."""

# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see THIRD_PARTY_LICENSES.md for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------
# Adapted through Pointcept / Point Transformer V3 (MIT License).
# Modified for SPARC3D / Hi3D to retain encoding only: the decode tables, the
# Hilbert curve and the batch-index field of the key are removed.
# --------------------------------------------------------

from __future__ import annotations

import torch


def _xyz2key_bitwise(x, y, z, depth):
    key = torch.zeros_like(x)
    for i in range(depth):
        mask = 1 << i
        key = (
            key
            | ((x & mask) << (2 * i + 2))
            | ((y & mask) << (2 * i + 1))
            | ((z & mask) << (2 * i + 0))
        )
    return key


_R256 = torch.arange(256, dtype=torch.int64)
_ZERO = torch.zeros(256, dtype=torch.int64)
_LUT_CPU = (
    _xyz2key_bitwise(_R256, _ZERO, _ZERO, 8),
    _xyz2key_bitwise(_ZERO, _R256, _ZERO, 8),
    _xyz2key_bitwise(_ZERO, _ZERO, _R256, 8),
)
_LUTS: dict[torch.device, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _lut(device: torch.device):
    if device not in _LUTS:
        _LUTS[device] = tuple(table.to(device) for table in _LUT_CPU)
    return _LUTS[device]


def xyz2key(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, depth: int = 16) -> torch.Tensor:
    """Interleave the bits of ``x``, ``y``, ``z`` into Morton keys (depth < 17)."""
    ex, ey, ez = _lut(x.device)
    x, y, z = x.long(), y.long(), z.long()
    mask = 255 if depth > 8 else (1 << depth) - 1
    key = ex[x & mask] | ey[y & mask] | ez[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = ex[(x >> 8) & mask] | ey[(y >> 8) & mask] | ez[(z >> 8) & mask]
        key = key16 << 24 | key
    return key


@torch.no_grad()
def encode(grid_coord: torch.Tensor, batch: torch.Tensor | None = None, depth: int = 16, order: str = "z"):
    """Encode grid coordinates as z-order codes, optionally keyed by batch.

    Args:
        grid_coord: Integer coordinates ``[N, 3]``.
        batch: Optional batch index per point, packed above the coordinate bits.
        depth: Bits per axis.
        order: ``"z"``, or ``"z-trans"`` to swap the first two axes first.
    """
    if order == "z":
        coords = grid_coord
    elif order == "z-trans":
        coords = grid_coord[:, [1, 0, 2]]
    else:
        raise ValueError(f"unsupported serialization order: {order!r}")
    code = xyz2key(coords[:, 0], coords[:, 1], coords[:, 2], depth=depth)
    if batch is not None:
        code = batch.long() << depth * 3 | code
    return code
