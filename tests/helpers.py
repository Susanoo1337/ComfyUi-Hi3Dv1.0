"""Small random models and meshes for CPU tests."""

import torch

from hi3d.transformer import Hi3DTransformer, TransformerConfig
from hi3d.vae import DecoderConfig, SparconvVAE


def icosphere(subdivisions=2, radius=1.0, center=(0.0, 0.0, 0.0)):
    """Octahedron subdivided ``subdivisions`` times and projected to a sphere."""
    vertices = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    faces = [[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]]
    for _ in range(subdivisions):
        cache = {}

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                vertices.append([(vertices[a][i] + vertices[b][i]) / 2 for i in range(3)])
                cache[key] = len(vertices) - 1
            return cache[key]

        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        faces = new_faces
    v = torch.tensor(vertices, dtype=torch.float64)
    v = v / v.norm(dim=1, keepdim=True) * radius + torch.tensor(center, dtype=torch.float64)
    return v.numpy(), torch.tensor(faces).numpy()


def init_random(module, std=0.05, seed=0):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if not param.dtype.is_floating_point:
                continue
            values = torch.randn(param.shape, generator=generator) * std
            if name.endswith(("norm1.weight", "norm2.weight", "weights", "gamma")) or ".norm" in name and name.endswith("weight"):
                values = values + 1.0
            param.copy_(values.to(param.dtype))
    return module


TINY_TRANSFORMER = TransformerConfig(
    in_channels=32,
    out_channels=32,
    model_channels=24,
    cond_channels=16,
    num_blocks=2,
    num_heads=2,
    mlp_ratio=2.0,
    num_io_res_blocks=2,
    io_block_channels=(8,),
)

TINY_DECODER = DecoderConfig(
    depths=(1, 1, 1),
    channels=(8, 16, 16),
    mid_depth=2,
    mid_num_head=2,
    patch_size=8,
    mlp_ratio=2.0,
)


def tiny_transformer(dtype=torch.float32):
    model = Hi3DTransformer(TINY_TRANSFORMER, dtype=dtype, compute_dtype=dtype)
    return init_random(model).eval()


def tiny_vae():
    return init_random(SparconvVAE(TINY_DECODER), std=0.2).eval()
