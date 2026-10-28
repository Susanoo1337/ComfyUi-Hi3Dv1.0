import json

import torch

from hi3d import loader
from hi3d.sparse import SparseTensor
from hi3d.transformer import LATENT_MEAN

from helpers import TINY_DECODER, TINY_TRANSFORMER, tiny_transformer, tiny_vae


def _forward(model, dtype=torch.float32):
    coords = torch.unique(torch.randint(0, 10, (120, 3), generator=torch.Generator().manual_seed(1)), dim=0).int()
    coords = torch.cat((torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords), dim=1)
    x = SparseTensor(torch.randn(coords.shape[0], 32, generator=torch.Generator().manual_seed(0)), coords)
    return model(x, torch.tensor([500.0]), torch.ones(1, 10, 16)).feats


def test_transformer_config_is_inferred_from_shapes():
    reference = tiny_transformer()
    sd = {"model.diffusion_model." + k: v for k, v in reference.state_dict().items()}
    model = loader.build_transformer(sd, compute_dtype=torch.float32)
    assert model.config == TINY_TRANSFORMER
    torch.testing.assert_close(_forward(model), _forward(reference))


def test_transformer_fp8_storage_runs_close_to_full_precision():
    reference = tiny_transformer()
    model = loader.build_transformer(
        reference.state_dict(), weight_dtype=torch.float8_e4m3fn, compute_dtype=torch.float32
    )
    assert model.blocks[0].mlp.mlp[0].weight.dtype == torch.float8_e4m3fn
    assert model.input_layer.weight.dtype == torch.float32
    out, expected = _forward(model), _forward(reference)
    assert (out - expected).abs().max() < 0.1 * expected.abs().max()


def test_metadata_overrides_latent_statistics():
    reference = tiny_transformer()
    stats = [0.5] * 32
    metadata = {"hi3d_config": json.dumps({"latent_mean": stats, "resolution": 128})}
    model = loader.build_transformer(reference.state_dict(), metadata=metadata, compute_dtype=torch.float32)
    assert model.config.latent_mean == tuple(stats)
    assert loader.build_transformer(reference.state_dict(), compute_dtype=torch.float32).config.latent_mean == LATENT_MEAN


def test_vae_config_is_inferred_and_release_config_is_read():
    reference = tiny_vae()
    sidecar = {"model": {"type": "SparconvVAE", "decoder_cfg": {"type": "PTv3MeshDecoder", "mid_num_head": 2, "patch_size": 8}}}
    model = loader.build_vae(reference.state_dict(), sidecar=sidecar)
    assert model.config == TINY_DECODER
    assert model.decoder.dec.dec2.up.upsample.inv_grid_coord.dtype == torch.int64


def test_wrong_file_is_rejected():
    reference = tiny_vae()
    try:
        loader.build_transformer(reference.state_dict())
    except ValueError as error:
        assert "not a Hi3D transformer" in str(error)
    else:
        raise AssertionError("a VAE state dict was accepted as a transformer")
