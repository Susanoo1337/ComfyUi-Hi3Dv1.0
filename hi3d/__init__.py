"""Pure-PyTorch Hi3D v1.0 inference: sparse flow transformer, sparse VAE, mesh ops.

This package has no hard dependency on ComfyUI, spconv, flash-attn or Triton. When
it runs inside ComfyUI, the node layer passes ComfyUI's casting layers and
attention in; standalone it falls back to plain PyTorch equivalents.
"""

__version__ = "1.0.0"
