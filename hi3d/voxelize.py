"""Surface voxelization of triangle meshes in PyTorch.

Ported from the Triton kernels of the reference release, which descend from
pytorch-voxelization (Zhihao Li). A voxel is occupied when its unit cube overlaps
a triangle by the separating-axis test; candidates are the voxels in each
triangle's bounding box, evaluated in fixed-size batches so any triangle size
works on any device.
"""

from __future__ import annotations

import logging

import torch

DEFAULT_PAD = 3
"""Default empty voxel margin per side."""

MIN_RESOLUTION = 2 * DEFAULT_PAD + 1
"""Smallest grid with a positive voxelization scale at :data:`DEFAULT_PAD`."""

MAX_RESOLUTION = 2048
"""Largest grid the 11-bit-per-axis voxel key can encode."""

_CANDIDATE_BATCH = 1 << 22
"""Triangle/voxel pairs tested per batch."""


def _axis_overlaps(v0, v1, v2, axis):
    p0 = (v0 * axis).sum(-1)
    p1 = (v1 * axis).sum(-1)
    p2 = (v2 * axis).sum(-1)
    lower = torch.minimum(torch.minimum(p0, p1), p2)
    upper = torch.maximum(torch.maximum(p0, p1), p2)
    radius = 0.5 * axis.abs().sum(-1)
    return (lower <= radius) & (upper >= -radius)


def _cross_axis(edge, which):
    zero = torch.zeros_like(edge[..., 0])
    ex, ey, ez = edge.unbind(-1)
    if which == 0:
        return torch.stack((zero, ez, -ey), -1)
    if which == 1:
        return torch.stack((-ez, zero, ex), -1)
    return torch.stack((ey, -ex, zero), -1)


def triangle_overlaps_voxel(a, b, c, voxel):
    """Separating-axis test of triangles ``(a, b, c)`` against unit voxels.

    Args:
        a, b, c: Triangle corners ``[M, 3]`` in voxel units.
        voxel: Integer voxel coordinates ``[M, 3]``; voxel ``v`` spans ``[v, v + 1]``.
    """
    center = voxel.to(a.dtype) + 0.5
    v0, v1, v2 = a - center, b - center, c - center
    overlaps = (
        (torch.minimum(torch.minimum(v0, v1), v2) <= 0.5)
        & (torch.maximum(torch.maximum(v0, v1), v2) >= -0.5)
    ).all(-1)
    edges = (v1 - v0, v2 - v1, v0 - v2)
    for edge in edges:
        for which in range(3):
            overlaps &= _axis_overlaps(v0, v1, v2, _cross_axis(edge, which))
    e0, e1 = edges[0], edges[1]
    normal = torch.cross(e0, e1, dim=-1)
    plane_offset = -(normal * v0).sum(-1)
    plane_radius = 0.5 * normal.abs().sum(-1)
    return overlaps & (plane_offset.abs() <= plane_radius)


def surface_voxels(vertices: torch.Tensor, faces: torch.Tensor, resolution: int) -> torch.Tensor:
    """Occupied voxels of a triangle soup already expressed in voxel units.

    Args:
        vertices: Float coordinates ``[V, 3]`` in voxel space.
        faces: Triangle indices ``[F, 3]``.
        resolution: Grid side length; voxels outside ``[0, resolution)`` are dropped.

    Returns:
        Unique int32 coordinates ``[N, 3]`` sorted by ``(z, y, x)``, the order of the
        reference kernel's ``z << 22 | y << 11 | x`` keys.
    """
    if not 1 <= resolution <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be in [1, {MAX_RESOLUTION}], got {resolution}")
    vertices = vertices.to(torch.float32)
    triangles = vertices[faces.long()]  # [F, 3, 3]
    low = triangles.min(dim=1).values.floor().to(torch.int64).clamp(min=0)
    high = triangles.max(dim=1).values.floor().to(torch.int64).clamp(max=resolution - 1)
    size = (high - low + 1).clamp(min=0)
    counts = size.prod(dim=-1)
    ends = torch.cumsum(counts, dim=0)
    total = int(ends[-1].item()) if len(ends) else 0
    starts = ends - counts

    keys = []
    for begin in range(0, total, _CANDIDATE_BATCH):
        index = torch.arange(begin, min(begin + _CANDIDATE_BATCH, total), device=vertices.device)
        face = torch.searchsorted(ends, index, right=True)
        local = index - starts[face]
        sx, sy = size[face, 0], size[face, 1]
        offset = torch.stack((local % sx, (local // sx) % sy, local // (sx * sy)), dim=-1)
        voxel = low[face] + offset
        tri = triangles[face]
        hit = triangle_overlaps_voxel(tri[:, 0], tri[:, 1], tri[:, 2], voxel)
        voxel = voxel[hit]
        keys.append((voxel[:, 2] << 22) | (voxel[:, 1] << 11) | voxel[:, 0])

    if not keys:
        return torch.empty((0, 3), dtype=torch.int32, device=vertices.device)
    unique = torch.unique(torch.cat(keys))
    mask = (1 << 11) - 1
    coords = torch.stack((unique & mask, (unique >> 11) & mask, (unique >> 22) & mask), dim=-1)
    return coords.to(torch.int32)


class GridFrame:
    """Uniform scale and offset placing a mesh inside a padded voxel grid.

    The bounding-box center (with an ``eps`` margin) maps to the grid center and the
    longest side spans ``resolution - 2 * pad - 2 * eps`` voxels.
    """

    def __init__(self, vertices: torch.Tensor, resolution: int, eps: float = 1e-6, pad: int = DEFAULT_PAD):
        self.resolution = int(resolution)
        low = vertices.min(dim=0).values - eps
        high = vertices.max(dim=0).values + eps
        self.center = (low + high) / 2.0
        self.scale = (self.resolution - pad * 2 - 2 * eps) / (high - low).max()

    def to_voxel(self, vertices: torch.Tensor) -> torch.Tensor:
        return (vertices - self.center) * self.scale + (self.resolution / 2.0)


def adaptive_vox(vertices, faces, resolution: int = 64, target_num: int = 80000, device="cpu"):
    """Shrink-to-fit surface voxelization.

    Voxelizes at ``int(resolution * scale)`` and, while more than ``target_num``
    voxels are occupied, multiplies ``scale`` by 0.95 and retries.

    Returns:
        ``(coords, frame)``: int32 coordinates ``[N, 3]`` and the :class:`GridFrame`
        of the accepted grid, which maps the input vertices into voxel units.
    """
    if target_num < 1:
        raise ValueError(f"target_num must be positive, got {target_num}")
    if resolution < MIN_RESOLUTION:
        raise ValueError(f"resolution must be at least {MIN_RESOLUTION}, got {resolution}")
    vertices = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(faces, dtype=torch.int64, device=device)

    scale = 1.0
    while True:
        current = int(resolution * scale)
        if current < MIN_RESOLUTION:
            raise ValueError(
                f"shrinking from resolution {resolution} reached {current}; "
                f"target_num={target_num} is unreachable for this mesh"
            )
        frame = GridFrame(vertices, current)
        coords = surface_voxels(frame.to_voxel(vertices), faces, current)
        if coords.shape[0] <= target_num:
            break
        scale *= 0.95

    if current != resolution:
        logging.warning(
            "Hi3D: voxelization settled at resolution %d instead of %d to fit %d voxels",
            current, resolution, target_num,
        )
    return coords, frame
