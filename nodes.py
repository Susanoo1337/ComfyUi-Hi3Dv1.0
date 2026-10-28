"""ComfyUI nodes for Hi3D v1.0 image-conditioned mesh refinement."""

from __future__ import annotations

import inspect
import logging
import os

import numpy as np
import torch
from typing_extensions import override

import comfy.model_management as mm
import comfy.model_patcher
import comfy.ops
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, Types, io

from .hi3d import loader
from .hi3d.conditioning import assemble_condition, encoder_inputs, release_position_interpolation, tensor_to_pil
from .hi3d.ops import autocast_off
from .hi3d.pipeline import OUTPUT_RESOLUTIONS, STRUCTURE_RESOLUTIONS, refine

log = logging.getLogger("Hi3D")

TransformerSocket = io.Custom("HI3D_TRANSFORMER")
VAESocket = io.Custom("HI3D_VAE")
ConditioningSocket = io.Custom("HI3D_CONDITIONING")

CATEGORY = "Hi3D"

_WEIGHT_DTYPES = {
    "default": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp8_e4m3fn": torch.float8_e4m3fn,
    "fp8_e4m3fn_fast": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
_VAE_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

AXES_GLTF = "Y-up (glTF, Trellis2, Pixal3D, Load 3D)"
AXES_HUNYUAN = "X/Z swapped (native Hunyuan3D VoxelToMesh)"
AXES_ZUP = "Z-up (Blender, raw TRELLIS)"


def _advanced() -> dict:
    """``advanced=True`` on ComfyUI versions whose widgets support it."""
    if "advanced" in inspect.signature(io.Int.Input.__init__).parameters:
        return {"advanced": True}
    return {}


def _new_patcher(model, load_device, offload_device):
    # The classic patcher moves whole modules; the custom sparse layers here hold
    # plain parameters, which it handles without per-op casting hooks.
    return comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)


class LoadedModel:
    """A loaded Hi3D module plus the ComfyUI patcher that owns its residency."""

    def __init__(self, model, name: str, load_device, offload_device, structure_resolution: int | None = None):
        self.model = model
        self.name = name
        self.structure_resolution = structure_resolution
        self.load_device = load_device
        self.patcher = _new_patcher(model, load_device, offload_device)

    def load(self, memory_required: int = 0):
        mm.load_models_gpu([self.patcher], memory_required=memory_required)
        return self.model


def _load_file(folder: str, name: str):
    path = folder_paths.get_full_path_or_raise(folder, name)
    sd, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    return path, sd, metadata or {}


def _compute_dtype(device, weight_choice: str) -> torch.dtype:
    if weight_choice == "bf16":
        return torch.bfloat16
    if weight_choice == "fp32":
        return torch.float32
    if mm.should_use_fp16(device):
        return torch.float16
    if mm.should_use_bf16(device):
        return torch.bfloat16
    return torch.float32


def _structure_resolution(path: str, sidecar: dict, metadata: dict) -> int | None:
    """Which ladder rung a transformer file was trained for, when it says so."""
    config = {**sidecar, **loader.metadata_config(metadata)}
    if "resolution" in config:
        return int(config["resolution"])
    name = os.path.basename(os.path.dirname(path)) + "/" + os.path.basename(path)
    if "1024" in name:
        return 128
    if "512" in name:
        return 64
    return None


class Hi3DTransformerLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Hi3DTransformerLoader",
            display_name="Load Hi3D Transformer",
            category=CATEGORY,
            description=(
                "Load a Hi3D sparse flow transformer from models/diffusion_models. Hi3D uses "
                "two: transformer_512 for the first pass and transformer_1024 for the "
                "1024 and 1536 passes. Nothing is downloaded."
            ),
            search_aliases=["hi3d", "sparc3d", "refine mesh", "transformer_512", "transformer_1024"],
            inputs=[
                io.Combo.Input("transformer_name", options=folder_paths.get_filename_list("diffusion_models")),
                io.Combo.Input(
                    "weight_dtype",
                    options=list(_WEIGHT_DTYPES),
                    default="default",
                    tooltip=(
                        "Storage precision of the transformer weights. 'default' keeps the file's "
                        "precision. fp8 halves VRAM; quantized checkpoints (scaled fp8, comfy_quant) "
                        "are detected automatically."
                    ),
                ),
            ],
            outputs=[TransformerSocket.Output(display_name="transformer")],
        )

    @classmethod
    def execute(cls, transformer_name, weight_dtype) -> io.NodeOutput:
        path, sd, metadata = _load_file("diffusion_models", transformer_name)
        sidecar = loader.read_sidecar_config(path)
        load_device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        compute_dtype = _compute_dtype(load_device, weight_dtype)

        sd = loader.normalize_state_dict(sd, loader.TRANSFORMER_MARKER)
        convert = getattr(comfy.utils, "convert_old_quants", None)
        if convert is not None:
            sd, metadata = convert(sd, "", metadata=metadata)
            metadata = metadata or {}
        requested = _WEIGHT_DTYPES[weight_dtype]
        if loader.is_quantized(sd):
            config = type("QuantConfig", (), {"quant_config": comfy.utils.detect_layer_quantization(sd, "")})
            ops = comfy.ops.pick_operations(None, compute_dtype, load_device=load_device, model_config=config)
            requested = None
        elif weight_dtype == "fp8_e4m3fn_fast":
            ops = comfy.ops.pick_operations(requested, compute_dtype, load_device=load_device, fp8_optimizations=True)
        else:
            ops = comfy.ops.manual_cast

        model = loader.build_transformer(
            sd,
            metadata=metadata,
            sidecar=sidecar,
            ops=ops,
            weight_dtype=requested,
            compute_dtype=compute_dtype,
            device=offload_device,
        )
        structure = _structure_resolution(path, sidecar, metadata)
        log.info("Hi3D: loaded %s (%s compute, structure %s)", transformer_name, compute_dtype, structure)
        return io.NodeOutput(LoadedModel(model, transformer_name, load_device, offload_device, structure))


class Hi3DVAELoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Hi3DVAELoader",
            display_name="Load Hi3D VAE",
            category=CATEGORY,
            description="Load the Hi3D sparse mesh VAE decoder from models/vae. Nothing is downloaded.",
            search_aliases=["hi3d", "sparc3d", "sparse vae"],
            inputs=[
                io.Combo.Input("vae_name", options=folder_paths.get_filename_list("vae")),
                io.Combo.Input(
                    "dtype",
                    options=list(_VAE_DTYPES),
                    default="fp32",
                    tooltip="Weight precision. The release decodes with fp32 weights; fp16/bf16 halve "
                    "the convolution memory at a small accuracy cost.",
                ),
            ],
            outputs=[VAESocket.Output(display_name="vae")],
        )

    @classmethod
    def execute(cls, vae_name, dtype) -> io.NodeOutput:
        path, sd, metadata = _load_file("vae", vae_name)
        load_device = mm.get_torch_device()
        offload_device = mm.vae_offload_device()
        model = loader.build_vae(
            sd,
            metadata=metadata,
            sidecar=loader.read_sidecar_config(path),
            ops=comfy.ops.manual_cast,
            dtype=_VAE_DTYPES[dtype],
            device=offload_device,
        )
        return io.NodeOutput(LoadedModel(model, vae_name, load_device, offload_device))


def _check_encoder(model, kind: str) -> None:
    config = getattr(model, "config", {}) or {}
    model_type = getattr(model, "model_type", None)
    if kind == "dino":
        if model_type != "dinov2" or config.get("hidden_size") != 1024 or config.get("num_hidden_layers") != 24:
            raise ValueError(
                "dino_model must be DINOv2-large (facebook/dinov2-large, no registers), loaded "
                f"with Load CLIP Vision; got a {model_type} model"
            )
    else:
        if (
            model_type != "clip_vision_model"
            or config.get("hidden_size") != 1024
            or config.get("image_size", 224) != 224
            or config.get("patch_size") != 14
        ):
            raise ValueError(
                "clip_model must be CLIP ViT-L/14 at 224px (openai/clip-vit-large-patch14, e.g. "
                f"clip_vision_l.safetensors), loaded with Load CLIP Vision; got a {model_type} model"
            )


class Hi3DConditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Hi3DConditioning",
            display_name="Hi3D Conditioning",
            category=CATEGORY,
            description=(
                "Encode the reference image with DINOv2-large and CLIP ViT-L/14. The subject is "
                "cropped to its mask, put on white and padded so it fills 1/1.2 of the frame, "
                "exactly as the release preprocesses it."
            ),
            search_aliases=["hi3d", "sparc3d", "image conditioning"],
            inputs=[
                io.ClipVision.Input("dino_model", tooltip="DINOv2-large from Load CLIP Vision."),
                io.ClipVision.Input("clip_model", tooltip="CLIP ViT-L/14 (224px) from Load CLIP Vision."),
                io.Image.Input("image", tooltip="Front view of the object the coarse mesh was made from."),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip="Foreground mask (1 = subject), e.g. from Remove Background. Without "
                    "one the whole frame is used. The mask output of Load Image is inverted "
                    "alpha: pass it through Invert Mask first.",
                ),
            ],
            outputs=[
                ConditioningSocket.Output(display_name="conditioning"),
                io.Image.Output(display_name="framed_image"),
            ],
        )

    @classmethod
    def execute(cls, dino_model, clip_model, image, mask=None) -> io.NodeOutput:
        _check_encoder(dino_model, "dino")
        _check_encoder(clip_model, "clip")
        mm.load_models_gpu([dino_model.patcher, clip_model.patcher])
        out_device = mm.intermediate_device()

        conds, previews = [], []
        for index in range(image.shape[0]):
            item_mask = None if mask is None else mask[min(index, mask.shape[0] - 1)]
            prepared, dino_pixels, clip_pixels = encoder_inputs(tensor_to_pil(image[index], item_mask))
            # The release runs both encoders in float32; ComfyUI's casting layers
            # follow the input dtype.
            with torch.no_grad(), autocast_off(dino_model.load_device.type):
                with release_position_interpolation(dino_model.model.embeddings):
                    tokens = dino_model.model(pixel_values=dino_pixels.to(dino_model.load_device))[0]
                pooled = clip_model.model.vision_model(pixel_values=clip_pixels.to(clip_model.load_device))[2]
            conds.append(assemble_condition(tokens, pooled).to(out_device))
            previews.append(torch.from_numpy(np.asarray(prepared, dtype=np.float32) / 255.0))

        size = max(p.shape[0] for p in previews)
        previews = [
            p if p.shape[0] == size else comfy.utils.common_upscale(p.movedim(-1, 0)[None], size, size, "lanczos", "disabled")[0].movedim(0, -1)
            for p in previews
        ]
        return io.NodeOutput({"cond": torch.cat(conds, dim=0)}, torch.stack(previews))


def _mesh_items(mesh):
    items = []
    counts_v = getattr(mesh, "vertex_counts", None)
    counts_f = getattr(mesh, "face_counts", None)
    for index in range(mesh.vertices.shape[0]):
        vertices, faces = mesh.vertices[index], mesh.faces[index]
        if counts_v is not None:
            vertices = vertices[: int(counts_v[index])]
            faces = faces[: int(counts_f[index])]
        items.append((vertices.detach().double().cpu().numpy(), faces.detach().long().cpu().numpy()))
    return items


def _to_model_axes(vertices: np.ndarray, axes: str) -> np.ndarray:
    if axes == AXES_HUNYUAN:
        return vertices[:, [2, 1, 0]]
    if axes == AXES_ZUP:
        return np.stack((vertices[:, 0], vertices[:, 2], -vertices[:, 1]), axis=-1)
    return vertices


def _from_model_axes(vertices: np.ndarray, faces: np.ndarray, axes: str):
    if axes == AXES_HUNYUAN:
        # A mirror: flip the winding so normals still face outwards.
        return vertices[:, [2, 1, 0]], faces[:, [0, 2, 1]]
    if axes == AXES_ZUP:
        return np.stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]), axis=-1), faces
    return vertices, faces


def _pack_meshes(items):
    vertices = [torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32)) for v, _ in items]
    faces = [torch.from_numpy(np.ascontiguousarray(f, dtype=np.int32)) for _, f in items]
    if len(items) == 1:
        return Types.MESH(vertices=vertices[0][None], faces=faces[0][None])
    packed_v = torch.zeros((len(items), max(v.shape[0] for v in vertices), 3))
    packed_f = torch.zeros((len(items), max(f.shape[0] for f in faces), 3), dtype=torch.int32)
    for i, (v, f) in enumerate(zip(vertices, faces)):
        packed_v[i, : v.shape[0]] = v
        packed_f[i, : f.shape[0]] = f
    return Types.MESH(
        vertices=packed_v,
        faces=packed_f,
        vertex_counts=torch.tensor([v.shape[0] for v in vertices]),
        face_counts=torch.tensor([f.shape[0] for f in faces]),
    )


class Hi3DRefineMesh(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        adv = _advanced()
        return io.Schema(
            node_id="Hi3DRefineMesh",
            display_name="Hi3D Refine Mesh",
            category=CATEGORY,
            description=(
                "Refine a coarse mesh into high-resolution geometry conditioned on the image. "
                "512 runs one pass, 1024 two and 1536 three. Output is geometry only."
            ),
            search_aliases=["hi3d", "sparc3d", "refine mesh", "image to 3d", "mesh detail"],
            inputs=[
                TransformerSocket.Input("transformer_512", tooltip="transformer_512 weights."),
                VAESocket.Input("vae"),
                ConditioningSocket.Input("conditioning"),
                io.Mesh.Input(
                    "mesh",
                    tooltip="Coarse mesh of the same object, e.g. from Trellis2, Pixal3D or "
                    "Hunyuan3D, or Load 3D + Get 3D Components.",
                ),
                io.Combo.Input(
                    "resolution",
                    options=[str(r) for r in OUTPUT_RESOLUTIONS],
                    default="1024",
                    tooltip="Nominal extraction grid. 1024 and 1536 need transformer_1024.",
                ),
                io.Int.Input("steps", default=25, min=1, max=200, tooltip="Scheduler updates per pass."),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True),
                io.Combo.Input(
                    "coarse_mesh_axes",
                    options=[AXES_GLTF, AXES_HUNYUAN, AXES_ZUP],
                    default=AXES_GLTF,
                    tooltip="Axis convention of the input mesh; the output uses the same one. "
                    "Hi3D sees the image as the object's front (+Z, Y up).",
                ),
                io.Combo.Input(
                    "output_frame",
                    options=["match input", "normalized"],
                    default="match input",
                    tooltip="'match input' places the result where the coarse mesh was, so it "
                    "lines up with texturing nodes. 'normalized' centers it with unit radius, "
                    "like the reference release.",
                ),
                TransformerSocket.Input("transformer_1024", optional=True, tooltip="transformer_1024 weights."),
                io.Int.Input(
                    "target_faces",
                    default=-1,
                    min=-1,
                    max=100_000_000,
                    tooltip="-1: release default (1M faces, 2M at 1536). 0: no simplification.",
                    **adv,
                ),
                io.Float.Input(
                    "remove_debris",
                    default=0.005,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="Drop parts holding less than this share of all faces (the largest part is kept).",
                    **adv,
                ),
                io.Combo.Input("scheduler", options=["heun", "euler"], default="heun", **adv),
            ],
            outputs=[io.Mesh.Output(display_name="mesh")],
        )

    @classmethod
    def execute(
        cls,
        transformer_512,
        vae,
        conditioning,
        mesh,
        resolution,
        steps,
        seed,
        coarse_mesh_axes,
        output_frame,
        transformer_1024=None,
        target_faces=-1,
        remove_debris=0.005,
        scheduler="heun",
    ) -> io.NodeOutput:
        resolution = int(resolution)
        passes = OUTPUT_RESOLUTIONS.index(resolution) + 1
        if passes > 1 and transformer_1024 is None:
            raise ValueError(f"resolution {resolution} needs transformer_1024; connect it or use 512")
        for slot, model, expected in (("transformer_512", transformer_512, 64), ("transformer_1024", transformer_1024, 128)):
            if model is not None and model.structure_resolution not in (None, expected):
                log.warning("Hi3D: %s is connected to the %s input; check the two are not swapped", model.name, slot)

        conds = conditioning["cond"]
        items = _mesh_items(mesh)
        count = max(len(items), conds.shape[0])
        device = mm.get_torch_device()
        pbar = comfy.utils.ProgressBar(count * passes * steps)

        def on_step():
            mm.throw_exception_if_processing_interrupted()
            pbar.update(1)

        def get_transformer(structure):
            model = transformer_512 if structure == STRUCTURE_RESOLUTIONS[0] else transformer_1024
            return model.load(memory_required=3 * 1024**3)

        def get_vae():
            # Decoded rows grow ~16x per pass; reserve room so ComfyUI evicts other models.
            return vae.load(memory_required=6 * 1024**3 * passes)

        outputs = []
        for index in range(count):
            vertices, faces = items[min(index, len(items) - 1)]
            result = refine(
                get_transformer=get_transformer,
                get_vae=get_vae,
                condition=conds[min(index, conds.shape[0] - 1)][None],
                vertices=_to_model_axes(vertices, coarse_mesh_axes),
                faces=faces,
                resolution=resolution,
                steps=steps,
                seed=(seed + index) & 0xFFFFFFFFFFFFFFFF,
                scheduler=scheduler,
                target_face_count=None if target_faces < 0 else target_faces,
                component_face_ratio=remove_debris,
                device=device,
                on_step=on_step,
            )
            out_vertices = result.vertices
            if output_frame == "match input":
                out_vertices = result.to_input.apply(out_vertices)
            outputs.append(_from_model_axes(out_vertices, result.faces, coarse_mesh_axes))
            log.info("Hi3D: refined mesh %d at %d with %d faces", index, resolution, len(result.faces))
        return io.NodeOutput(_pack_meshes(outputs))


class Hi3DExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [Hi3DTransformerLoader, Hi3DVAELoader, Hi3DConditioning, Hi3DRefineMesh]


async def comfy_entrypoint() -> Hi3DExtension:
    return Hi3DExtension()
