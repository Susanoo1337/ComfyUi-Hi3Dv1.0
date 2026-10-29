"""ComfyUI-Hi3D: Hi3D v1.0 image-conditioned mesh refinement nodes for ComfyUI."""

# ComfyUI imports this folder as a package. Tools that import the file on its own
# (pytest collecting the repository root) get an empty module instead.
if __package__:
    from .nodes import comfy_entrypoint

    __all__ = ["comfy_entrypoint"]
