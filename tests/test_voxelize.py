import torch

from hi3d.voxelize import GridFrame, adaptive_vox, surface_voxels

from helpers import icosphere


def test_surface_voxels_cover_the_surface_and_nothing_else():
    vertices, faces = icosphere(3, radius=1.0)
    vertices = torch.from_numpy(vertices).float()
    faces = torch.from_numpy(faces)
    frame = GridFrame(vertices, 24)
    voxel_vertices = frame.to_voxel(vertices)
    coords = surface_voxels(voxel_vertices, faces, 24)

    occupied = {tuple(c) for c in coords.tolist()}
    # Every point on every triangle lies in an occupied voxel.
    generator = torch.Generator().manual_seed(0)
    weights = torch.rand(len(faces), 64, 3, generator=generator)
    weights = weights / weights.sum(-1, keepdim=True)
    points = torch.einsum("fsk,fkd->fsd", weights, voxel_vertices[faces.long()]).reshape(-1, 3)
    assert all(tuple(p) in occupied for p in points.floor().int().tolist())

    # And every occupied voxel touches the surface: its center lies within half a
    # voxel diagonal of the (convex, slightly inscribed) sphere.
    centers = coords.float() + 0.5
    radius = (centers - 12.0).norm(dim=1)
    sphere = float(frame.scale)
    assert (radius - sphere).abs().max() <= 3**0.5 / 2 + 0.05


def test_coordinates_are_sorted_like_the_reference_keys():
    vertices, faces = icosphere(2)
    coords = surface_voxels(GridFrame(torch.from_numpy(vertices).float(), 16).to_voxel(torch.from_numpy(vertices).float()), torch.from_numpy(faces), 16)
    keys = (coords[:, 2].long() << 22) | (coords[:, 1].long() << 11) | coords[:, 0].long()
    assert torch.all(keys[1:] > keys[:-1])


def test_adaptive_vox_shrinks_to_budget():
    vertices, faces = icosphere(3)
    full, _ = adaptive_vox(vertices, faces, resolution=32, target_num=10**9)
    small, frame = adaptive_vox(vertices, faces, resolution=32, target_num=len(full) // 2)
    assert len(small) <= len(full) // 2
    assert frame.resolution < 32
