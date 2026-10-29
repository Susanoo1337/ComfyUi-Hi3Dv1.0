"""Batch-1 Hi3D refinement ladder: sample, decode, extract, revoxelize.

Ported from ``Sparc3DPipeline`` of the reference release. Model residency is left
to the caller through the ``get_transformer`` / ``get_vae`` callbacks, so ComfyUI's
memory manager decides what stays on the GPU between passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .extract import point_to_mesh
from .fill_hole import fill_mesh_holes
from .ops import autocast_fp16
from .postprocess import postprocess_mesh, validate_mesh
from .scheduler import SCHEDULERS, apply_effective_guidance
from .sparse import SparseTensor
from .voxelize import adaptive_vox

STRUCTURE_RESOLUTIONS = (64, 128, 192)
"""Sparse structure resolution of each rung of the refinement ladder."""

OUTPUT_RESOLUTIONS = (512, 1024, 1536)
"""Nominal extraction resolution produced by each rung."""

PASS_BUDGETS = (80_000, 90_000, 110_000)
"""Occupied-voxel budget adaptive voxelization targets at each rung."""

TARGET_FACE_COUNTS = (1_000_000, 1_000_000, 2_000_000)
"""Default simplification target per output resolution."""

DECODE_FEATURE_CEILING = 12_000_000
"""Decoded rows above which a further pass would exhaust device memory."""

COMPONENT_FACE_RATIO = 0.005
"""Connected components holding a smaller share of faces are dropped as debris."""

LATENT_CHANNELS = 32


@dataclass
class Affine:
    """Uniform scale and translation: ``parent = child * scale + offset``."""

    scale: float = 1.0
    offset: np.ndarray = None

    def __post_init__(self):
        if self.offset is None:
            self.offset = np.zeros(3)

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points * self.scale + self.offset

    def then(self, parent: "Affine") -> "Affine":
        """Compose: map child -> self.parent frame -> ``parent``'s parent frame."""
        return Affine(self.scale * parent.scale, self.offset * parent.scale + parent.offset)


@dataclass
class RefineResult:
    vertices: np.ndarray
    """Unit-radius, mean-centered vertices ``[V, 3]`` (the reference output)."""
    faces: np.ndarray
    """Triangle indices ``[F, 3]``."""
    resolution: int
    """Nominal extraction resolution reached."""
    to_input: Affine
    """Maps :attr:`vertices` back into the coarse mesh's coordinate frame."""


def resolve_generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return generator


def _guided_velocity(transformer, sample, timestep, condition, negative_condition):
    """Predict one flow velocity, guided only from time 0.7 upwards."""
    model_timestep = (1000.0 * timestep).reshape(1)
    positive = transformer(sample, model_timestep, condition)
    if float(timestep) < 0.7:
        return positive.feats
    negative = transformer(sample, model_timestep, negative_condition)
    return apply_effective_guidance(positive.feats, negative.feats, timestep)


@torch.inference_mode()
def sample_latents(
    transformer,
    condition: torch.Tensor,
    coords: torch.Tensor,
    *,
    steps: int,
    generator: torch.Generator,
    scheduler: str = "heun",
    device: torch.device,
    on_step: Callable[[], None] | None = None,
) -> SparseTensor:
    """Sample VAE-space latents on ``coords`` ``[N, 3]`` with flow matching."""
    coords = coords.to(device=device, dtype=torch.int32)
    batched = torch.cat((torch.zeros((coords.shape[0], 1), dtype=torch.int32, device=device), coords), dim=1)
    noise = torch.randn((coords.shape[0], LATENT_CHANNELS), generator=generator, device="cpu", dtype=torch.float32)
    sample = SparseTensor(noise.to(device), batched)
    condition = condition.to(device)
    negative = torch.zeros_like(condition)

    integrator = SCHEDULERS[scheduler]()
    timesteps = integrator.set_timesteps(steps, num_tokens=coords.shape[0], device=device)
    for sigma_hat in timesteps:
        velocity = _guided_velocity(transformer, sample, sigma_hat, condition, negative)
        if integrator.order == 2:
            predicted, next_sigma = integrator.predict(velocity, sigma_hat, sample.feats)
            second = _guided_velocity(transformer, sample.replace(predicted), next_sigma, condition, negative)
            updated = integrator.step(velocity, sigma_hat, sample.feats, second_output=second)
        else:
            updated = integrator.step(velocity, sigma_hat, sample.feats)
        sample = sample.replace(updated)
        if on_step is not None:
            on_step()

    mean, std = transformer.latent_stats(sample.feats)
    return sample.replace(sample.feats * std + mean)


@torch.inference_mode()
def decode_and_extract(vae, latents: SparseTensor, structure_resolution: int, device: torch.device):
    """Decode latents and extract a mesh in the decoder's normalized frame.

    Returns:
        ``(vertices, faces, decoded_rows, to_grid)`` where ``to_grid`` maps mesh
        vertices into the voxel units of the structure grid the latents live on.
    """
    grid_coord = latents.coords[:, 1:].int()
    with autocast_fp16(device.type):
        decoded = vae.decode(grid_coord, latents.feats.float(), structure_resolution)
    rows = int(decoded.feat.shape[0])

    # grid_to_coord maps fine-grid positions p to (p - c) / s with c, s taken from
    # the structure-level coordinates, and a fine voxel is 1 / upscale of a
    # structure voxel, so structure units are (mesh * s + c) / upscale.
    matrix = decoded.matrix.detach().double().cpu().numpy()
    scale = 1.0 / matrix[0, 0]
    center = -matrix[:3, 3] * scale
    upscale = decoded.resolution / structure_resolution
    to_grid = Affine(scale / upscale, center / upscale)

    mesh = fill_mesh_holes(point_to_mesh(decoded))
    del decoded
    if not mesh.success:
        raise RuntimeError("Hi3D: mesh extraction produced an empty isosurface")
    return mesh.vertices.detach().float(), mesh.faces.detach().long(), rows, to_grid


@torch.inference_mode()
def refine(
    *,
    get_transformer: Callable[[int], torch.nn.Module],
    get_vae: Callable[[], torch.nn.Module],
    condition: torch.Tensor,
    vertices,
    faces,
    resolution: int = 1024,
    steps: int = 25,
    seed: int = 0,
    scheduler: str = "heun",
    target_face_count: int | None = None,
    component_face_ratio: float = COMPONENT_FACE_RATIO,
    device: torch.device | str = "cuda",
    on_step: Callable[[], None] | None = None,
) -> RefineResult:
    """Refine a coarse mesh through the requested resolution.

    Args:
        get_transformer: Returns the transformer for a structure resolution (64 uses
            ``transformer_512``; 128 and 192 use ``transformer_1024``), on ``device``.
        get_vae: Returns the sparse VAE on ``device``.
        condition: Image tokens ``[1, 2917, 2048]``.
        vertices: Coarse mesh vertices ``[V, 3]`` in the model frame.
        faces: Coarse mesh triangles ``[F, 3]``.
        resolution: 512, 1024 or 1536; runs one, two or three passes.
        steps: Scheduler updates per pass.
        seed: Seed of the local CPU noise generator.
        scheduler: ``"heun"`` (released default) or ``"euler"``.
        target_face_count: Simplification target; ``None`` uses the release default
            for ``resolution`` and ``0`` disables simplification.
        component_face_ratio: Debris filter threshold; ``0`` keeps every component.
        device: Execution device.
        on_step: Called after every scheduler update (progress and interruption).
    """
    if resolution not in OUTPUT_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {OUTPUT_RESOLUTIONS}, got {resolution}")
    if steps <= 0:
        raise ValueError("steps must be positive")
    if scheduler not in SCHEDULERS:
        raise ValueError(f"scheduler must be one of {sorted(SCHEDULERS)}")
    device = torch.device(device)
    target_index = OUTPUT_RESOLUTIONS.index(resolution)
    if target_face_count is None:
        target_face_count = TARGET_FACE_COUNTS[target_index]

    generator = resolve_generator(seed)
    mesh_vertices, mesh_faces = validate_mesh(vertices, faces)
    mesh_vertices = torch.from_numpy(mesh_vertices)
    mesh_faces = torch.from_numpy(mesh_faces)
    # Maps the current mesh's frame back to the coarse input's frame.
    to_input = Affine()

    for ladder_index in range(target_index + 1):
        structure = STRUCTURE_RESOLUTIONS[ladder_index]
        coords, grid = adaptive_vox(
            mesh_vertices, mesh_faces, resolution=structure, target_num=PASS_BUDGETS[ladder_index], device=device
        )
        # Voxel units -> current mesh frame: v = (u - R / 2) / s + c.
        grid_scale = float(grid.scale)
        grid_center = grid.center.double().cpu().numpy()
        from_grid = Affine(1.0 / grid_scale, grid_center - (grid.resolution / 2.0) / grid_scale)

        latents = sample_latents(
            get_transformer(structure),
            condition,
            coords,
            steps=steps,
            generator=generator,
            scheduler=scheduler,
            device=device,
            on_step=on_step,
        )
        mesh_vertices, mesh_faces, rows, to_grid = decode_and_extract(get_vae(), latents, structure, device)
        del latents
        to_input = to_grid.then(from_grid).then(to_input)

        if ladder_index < target_index and rows > DECODE_FEATURE_CEILING:
            raise RuntimeError(
                f"Hi3D: decoded {rows} rows, over the {DECODE_FEATURE_CEILING} ceiling for "
                "another pass; use a lower resolution"
            )

    out_vertices, out_faces, center, radius = postprocess_mesh(
        mesh_vertices.cpu().numpy(),
        mesh_faces.cpu().numpy(),
        target_face_count=int(target_face_count),
        component_face_ratio=float(component_face_ratio),
    )
    to_input = Affine(radius, center).then(to_input)
    return RefineResult(out_vertices, out_faces, resolution, to_input)
