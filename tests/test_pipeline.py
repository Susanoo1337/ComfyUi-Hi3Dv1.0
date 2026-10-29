import numpy as np
import pytest
import torch

import hi3d.pipeline as pipeline
from hi3d.sparse import SparseTensor
from hi3d.sparse_cube import CUBE_CORNERS
from hi3d.vae import Point

from helpers import icosphere, tiny_transformer, tiny_vae


class SphereVAE:
    """Stands in for the decoder: emits the SDF of a known sphere on a 4x finer grid."""

    upscale = 4

    def __init__(self, center, radius):
        self.center = torch.as_tensor(center, dtype=torch.float64)
        self.radius = float(radius)

    def decode(self, grid_coord, feats, resolution):
        point = Point(
            {
                "grid_coord": grid_coord.int(),
                "feat": feats.float(),
                "sdf": None,
                "deform": None,
                "resolution": resolution,
                "offset": torch.tensor([grid_coord.shape[0]]),
            }
        )
        point.initialization()
        k = self.upscale
        offsets = torch.stack(torch.meshgrid(*(torch.arange(k),) * 3, indexing="ij"), -1).reshape(-1, 3)
        fine = (grid_coord.long()[:, None] * k + offsets[None]).reshape(-1, 3)
        corners = (fine[:, None] + CUBE_CORNERS.long()[None]).double() / k
        sdf = (corners - self.center).norm(dim=-1) - self.radius
        keep = sdf.abs().min(dim=1).values < 2.0 / k
        feat = torch.cat((sdf[keep] * k, torch.zeros(int(keep.sum()), 24, dtype=torch.float64)), dim=1).clamp(-1, 1)
        batch = torch.zeros(int(keep.sum()), dtype=torch.long)
        return point.clone(
            feat=feat.float(), grid_coord=fine[keep], resolution=resolution * k, batch=batch, offset=torch.tensor([len(batch)])
        )


@pytest.fixture
def small_ladder(monkeypatch):
    monkeypatch.setattr(pipeline, "STRUCTURE_RESOLUTIONS", (16, 32, 48))
    monkeypatch.setattr(pipeline, "PASS_BUDGETS", (10**6, 10**6, 10**6))


@pytest.mark.parametrize("passes", [1, 2])
def test_refine_returns_mesh_in_the_coarse_frame(small_ladder, monkeypatch, passes):
    center, radius = np.array([0.3, -1.2, 2.0]), 1.7
    vertices, faces = icosphere(3, radius=radius, center=center)
    transformer = tiny_transformer()

    # The stub decoder has to know where the sphere sits on each pass's grid. Pass 1
    # voxelizes the input; later passes voxelize the previous pass's output, whose
    # bounding box gives the sphere in that mesh's own frame.
    sphere = {"center": torch.as_tensor(center, dtype=torch.float32), "radius": radius}
    real_vox, real_decode = pipeline.adaptive_vox, pipeline.decode_and_extract

    def tracking_vox(v, f, resolution, target_num, device):
        coords, frame = real_vox(v, f, resolution=resolution, target_num=target_num, device=device)
        sphere["frame"] = frame
        return coords, frame

    def tracking_decode(vae, latents, structure, device):
        out = real_decode(vae, latents, structure, device)
        v = out[0].double()
        sphere["center"] = ((v.max(0).values + v.min(0).values) / 2).float()
        sphere["radius"] = float((v.max(0).values - v.min(0).values).max() / 2)
        return out

    def get_vae():
        frame = sphere["frame"]
        voxel_center = (sphere["center"] - frame.center) * frame.scale + frame.resolution / 2.0
        return SphereVAE(voxel_center.double(), sphere["radius"] * float(frame.scale))

    monkeypatch.setattr(pipeline, "adaptive_vox", tracking_vox)
    monkeypatch.setattr(pipeline, "decode_and_extract", tracking_decode)
    steps_taken = []
    result = pipeline.refine(
        get_transformer=lambda structure: transformer,
        get_vae=get_vae,
        condition=torch.randn(1, 10, 16),
        vertices=vertices,
        faces=faces,
        resolution=pipeline.OUTPUT_RESOLUTIONS[passes - 1],
        steps=2,
        seed=1,
        target_face_count=0,
        device="cpu",
        on_step=lambda: steps_taken.append(1),
    )

    assert len(steps_taken) == 2 * passes
    normalized = result.vertices
    assert abs(np.linalg.norm(normalized - normalized.mean(0), axis=1).max() - 1.0) < 1e-6

    restored = result.to_input.apply(normalized)
    distance = np.linalg.norm(restored - center, axis=1)
    assert np.abs(distance - radius).max() < 0.05 * radius


def test_sampling_is_seeded():
    transformer = tiny_transformer()
    coords = torch.unique(torch.randint(0, 12, (200, 3)), dim=0)
    condition = torch.randn(1, 10, 16)
    a = pipeline.sample_latents(transformer, condition, coords, steps=2, generator=pipeline.resolve_generator(3), device=torch.device("cpu"))
    b = pipeline.sample_latents(transformer, condition, coords, steps=2, generator=pipeline.resolve_generator(3), device=torch.device("cpu"))
    assert isinstance(a, SparseTensor)
    torch.testing.assert_close(a.feats, b.feats)
    assert torch.isfinite(a.feats).all()


def test_real_decoder_runs():
    vae = tiny_vae()
    coords = torch.unique(torch.randint(0, 16, (400, 3)), dim=0).int()
    decoded = vae.decode(coords, torch.randn(coords.shape[0], 32), 16)
    assert decoded.feat.shape[1] == 32 and decoded["sdf_sign"].shape[1] == 8
    assert decoded.resolution == 16 * 4
    assert torch.isfinite(decoded.feat).all()
