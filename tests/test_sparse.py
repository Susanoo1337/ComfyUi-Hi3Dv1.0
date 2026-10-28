import torch
import torch.nn.functional as F

from hi3d.sparse import SparseConv3d, SparseDownsample, SparseTensor, SparseUpsample, SubMConv3d, neighbor_map


def _random_coords(count, size, batch=0, seed=0):
    generator = torch.Generator().manual_seed(seed)
    grid = torch.randperm(size**3, generator=generator)[:count]
    xyz = torch.stack((grid // (size * size), (grid // size) % size, grid % size), dim=-1)
    return torch.cat((torch.full((count, 1), batch), xyz), dim=1).int()


def test_submanifold_conv_matches_dense_conv3d():
    torch.manual_seed(0)
    size, cin, cout = 9, 5, 7
    coords = _random_coords(300, size)
    feats = torch.randn(coords.shape[0], cin, dtype=torch.float64)
    conv = SubMConv3d(cin, cout, 3, dtype=torch.float64)
    torch.nn.init.normal_(conv.weight)
    torch.nn.init.normal_(conv.bias)

    out = conv.convolve(feats, neighbor_map(coords, 3))

    dense = torch.zeros(1, cin, size, size, size, dtype=torch.float64)
    x, y, z = coords[:, 1].long(), coords[:, 2].long(), coords[:, 3].long()
    dense[0, :, x, y, z] = feats.t()
    # spconv's [out, kx, ky, kz, in] layout is a dense [out, in, kx, ky, kz] correlation.
    expected = F.conv3d(dense, conv.weight.permute(0, 4, 1, 2, 3), conv.bias, padding=1)
    torch.testing.assert_close(out, expected[0, :, x, y, z].t())


def test_batches_do_not_interact():
    coords = torch.cat((_random_coords(50, 5, batch=0), _random_coords(50, 5, batch=1)))
    feats = torch.randn(coords.shape[0], 3, dtype=torch.float64)
    conv = SubMConv3d(3, 4, 3, dtype=torch.float64)
    torch.nn.init.normal_(conv.weight)
    torch.nn.init.zeros_(conv.bias)
    together = conv.convolve(feats, neighbor_map(coords, 3))
    alone = conv.convolve(feats[50:], neighbor_map(coords[50:], 3))
    torch.testing.assert_close(together[50:], alone)


def test_conv_cache_and_resampling_round_trip():
    coords = _random_coords(200, 8)
    x = SparseTensor(torch.randn(coords.shape[0], 4), coords)
    conv = SparseConv3d(4, 4)
    torch.nn.init.normal_(conv.conv.weight)
    torch.nn.init.zeros_(conv.conv.bias)
    conv(x)
    assert x.get_spatial_cache("subm3") is not None
    down = SparseDownsample(2)(x)
    assert down.coords[:, 1:].max() <= 3
    up = SparseUpsample(2)(down)
    assert torch.equal(up.coords, x.coords)
    # Each parent holds sum / (children + 1): the reference's scatter "mean" counts
    # its zero initial value, and the released weights expect that. Upsampling
    # hands the parent value back to every child.
    parent = (x.coords[:, 1:].long() // 2) @ torch.tensor([64, 8, 1])
    for code in parent.unique()[:10]:
        children = parent == code
        expected = x.feats[children].sum(0) / (int(children.sum()) + 1)
        torch.testing.assert_close(up.feats[children], expected.expand(int(children.sum()), -1))
