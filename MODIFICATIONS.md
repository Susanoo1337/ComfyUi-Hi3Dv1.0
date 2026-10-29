# Modifications to Hi3D v1.0

This repository is an unofficial ComfyUI port of the
[Hi3D v1.0 source release](https://github.com/hi3d-official/Hi3Dv1.0). It is not
an official Hi3D product and is not endorsed by OCEANO HK HOLDING LIMITED. As
Section 4 of the [Hi3D Open License Agreement](LICENSE) requires, this file
identifies what was changed. `LICENSE`, `NOTICE`, `AUTHORS`,
`THIRD_PARTY_LICENSES.md` and `hi3d/flexicubes/LICENSE.txt` are kept unchanged;
the per-file provenance headers name each file's upstream origin.

No model weights are modified or redistributed. `tools/convert_hi3d.py` can
re-encode a locally downloaded copy of the released weights (precision change,
single-file packaging); files it produces are modified Hi3D weights.

## Modified files

| File in this port | Derived from (upstream release) | Change |
| --- | --- | --- |
| `hi3d/transformer.py` | `sparc3d/models/transformers/*.py`, `sparc3d/nn/{norm,rope,utils}.py`, `sparc3d/nn/sparse/{attention,transformer,linear,nonlinearity}/*` | One module; layers built from pluggable casting operations (ComfyUI `operations`); flash-attn replaced by PyTorch / ComfyUI attention; precision handled by explicit storage and compute dtypes instead of `convert_to_fp16`. Architecture, parameter names and numerics are unchanged. |
| `hi3d/sparse.py` | `sparc3d/nn/sparse/{basic,conv,spatial}.py`, `sparc3d/ops/sparse_backend.py` | spconv replaced by a PyTorch gather-GEMM submanifold convolution with the same weight layout and kernel-offset convention. |
| `hi3d/vae.py` | `sparc3d/models/autoencoders/*.py`, `sparc3d/structures/*.py`, `sparc3d/utils/{misc,precision}.py` | One module; spconv and flash-attn replaced as above; the JSON config builder replaced by shape inference in `hi3d/loader.py`. |
| `hi3d/serialization.py` | `sparc3d/utils/serialization/*.py` | One module; same z-order codes. |
| `hi3d/voxelize.py` | `sparc3d/ops/voxelize.py`, `sparc3d/ops/triton/voxelize.py` | Triton kernels re-expressed as batched PyTorch; same overlap test and voxel order; runs on any device. Returns the grid transform so the pipeline can map results back. |
| `hi3d/pipeline.py` | `sparc3d/pipelines/sparc3d_pipeline.py` | Model residency moved to callbacks (ComfyUI memory management); tracks the affine transform of every pass so output can be placed in the input mesh's frame. Sampling, guidance, ladder and budgets unchanged. |
| `hi3d/loader.py` | `sparc3d/pipelines/loader.py`, `sparc3d/models/autoencoders/config.py` | Loads single files picked by the user; no Hugging Face access; architecture inferred from tensor shapes; fp16/bf16/fp8 storage and ComfyUI quantized checkpoints. |
| `hi3d/conditioning.py` | `sparc3d/models/conditioners/image.py` | Encoders are ComfyUI CLIP Vision models supplied by the user instead of `transformers` downloads; same preprocessing and token assembly. |
| `hi3d/image.py` | `sparc3d/utils/image.py` | `rembg` helper removed. |
| `hi3d/postprocess.py` | `sparc3d/sparcubes/postprocess.py` | `trimesh` replaced by NumPy/SciPy with the same component rule; also returns the normalization it applied. |
| `hi3d/scheduler.py` | `sparc3d/schedulers/*.py` | Merged into one module; behavior unchanged. |
| `hi3d/extract.py`, `hi3d/fill_hole.py`, `hi3d/sparse_cube.py` | `sparc3d/sparcubes/*.py` | Import paths; device-agnostic autocast guard; row deduplication through packed integer keys (same result, far less memory). |
| `hi3d/flexicubes/flexicubes.py` | `sparc3d/ops/flexicubes/flexicubes.py` | Edge deduplication through packed integer keys, marked "Hi3D modification" in the file. |
| `hi3d/flexicubes/__init__.py` | `sparc3d/ops/flexicubes/__init__.py` | Import path. |
| `README.md`, `pyproject.toml` | same files | Rewritten for the ComfyUI node pack. |

## Added files

`__init__.py`, `nodes.py` (ComfyUI nodes), `hi3d/__init__.py`, `hi3d/ops.py`,
`tools/convert_hi3d.py`, `example_workflows/`, `assets/workflow-refine-mesh.png`,
`tests/`, `requirements.txt` and this file.

## Removed files

`sparc3d/` (replaced by `hi3d/`), `demo/`, `docker/`, `docs/installation.md`,
`docs/pipeline.md`, `docs/reproduction.md`, `README.zh-CN.md`, `CONTRIBUTING.md`
and `MANIFEST.in`. They describe the standalone library and its CUDA-only stack,
which this port does not use.
