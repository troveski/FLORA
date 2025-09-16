import os
import random
import sys
import tempfile
from typing import Sequence, Mapping, Any, Union

import torch
from PIL import Image

# Set custom output directory (only for this script)
import folder_paths
folder_paths.set_output_directory("/home/ai4ar/alvaro/ComfyUI/cars_augmented")

# ───────────────────────────────
# Generate RGBA image with alpha=0 in regions matching class 9 from YOLO labels
def create_alpha_mask_from_yolo(image_path: str, label_path: str, target_class: int = 9) -> Image.Image:
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    px = img.load()

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            if int(cls) != target_class:
                continue
            xmin = int((xc - bw / 2) * w)
            ymin = int((yc - bh / 2) * h)
            xmax = int((xc + bw / 2) * w)
            ymax = int((yc + bh / 2) * h)
            for y in range(max(0, ymin), min(h, ymax)):
                for x in range(max(0, xmin), min(w, xmax)):
                    r, g, b, _ = px[x, y]
                    px[x, y] = (r, g, b, 0)  # Set transparency
    return img
# ───────────────────────────────

# ───── CONFIGURATION ─────
image_normal = "/home/ai4ar/alvaro/ComfyUI/car_test/car1.jpg"
label_path   = "/home/ai4ar/alvaro/ComfyUI/car_test/car1.txt"

# Create temporary masked image
masked_img = create_alpha_mask_from_yolo(image_normal, label_path)
tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
masked_img.save(tmp_file.name)
img_with_mask_path = tmp_file.name
print(f"Temporary mask saved at: {img_with_mask_path}")
# ─────────────────────────

# ───── ComfyUI utilities ─────
def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]

def find_path(name: str, path: str = None) -> str:
    path = path or os.getcwd()
    if name in os.listdir(path):
        return os.path.join(path, name)
    parent = os.path.dirname(path)
    return None if parent == path else find_path(name, parent)

def add_comfyui_directory_to_sys_path():
    comfy = find_path("ComfyUI")
    if comfy and os.path.isdir(comfy):
        sys.path.append(comfy)

def add_extra_model_paths():
    try:
        from main import load_extra_path_config
    except ImportError:
        from utils.extra_config import load_extra_path_config
    extra = find_path("extra_model_paths.yaml")
    if extra:
        load_extra_path_config(extra)

add_comfyui_directory_to_sys_path()
add_extra_model_paths()

def import_custom_nodes():
    import asyncio, execution, server
    from nodes import init_extra_nodes
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)
    init_extra_nodes()

from nodes import NODE_CLASS_MAPPINGS
# ────────────────────────────────

# ───── Main workflow ─────
def main():
    import_custom_nodes()

    with torch.inference_mode():
        # Load models
        dualclip = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        clip_pair = dualclip.load_clip(
            clip_name1="t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default"
        )
        text_enc = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        prompt_pos = text_enc.encode(text="A car", clip=get_value_at_index(clip_pair, 0))
        prompt_neg = text_enc.encode(text="", clip=get_value_at_index(clip_pair, 0))

        vae = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("diffusion_pytorch_model.safetensors")
        unet = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet("flux1-dev.safetensors", weight_dtype="fp8_e5m2")

        sampler_name = NODE_CLASS_MAPPINGS["KSamplerSelect"]().get_sampler("euler")
        noise = NODE_CLASS_MAPPINGS["RandomNoise"]().get_noise(noise_seed=random.randint(1, 2 ** 64))

        # Load input image with alpha mask
        load_img = NODE_CLASS_MAPPINGS["LoadAndResizeImage"]()
        img_and_mask = load_img.load_image(
            image=img_with_mask_path,
            resize=True, width=640, height=640,
            repeat=1, keep_proportion=True, divisible_by=1,
            mask_channel="alpha", background_color=""
        )

        # Blur the alpha mask
        mask_blur = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]().doit(
            kernel_size=10, sigma=10,
            mask=get_value_at_index(img_and_mask, 1)
        )

        # Set up guidance
        guidance = NODE_CLASS_MAPPINGS["FluxGuidance"]().append(
            guidance=3.5,
            conditioning=get_value_at_index(prompt_pos, 0)
        )

        # Inpaint conditioning
        inpaint_cond = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]().encode(
            noise_mask=True,
            positive=get_value_at_index(guidance, 0),
            negative=get_value_at_index(prompt_neg, 0),
            vae=get_value_at_index(vae, 0),
            pixels=get_value_at_index(img_and_mask, 0),
            mask=get_value_at_index(mask_blur, 0)
        )

        # Diffusion and generation
        diff_model = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]().apply(
            model=get_value_at_index(unet, 0)
        )
        guider = NODE_CLASS_MAPPINGS["BasicGuider"]().get_guider(
            model=get_value_at_index(diff_model, 0),
            conditioning=get_value_at_index(inpaint_cond, 0)
        )
        sigmas = NODE_CLASS_MAPPINGS["BasicScheduler"]().get_sigmas(
            scheduler="simple", steps=15, denoise=1,
            model=get_value_at_index(unet, 0)
        )
        samples = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]().sample(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guider, 0),
            sampler=get_value_at_index(sampler_name, 0),
            sigmas=get_value_at_index(sigmas, 0),
            latent_image=get_value_at_index(inpaint_cond, 2)
        )

        # Decode and save output image
        decoded = NODE_CLASS_MAPPINGS["VAEDecode"]().decode(
            samples=get_value_at_index(samples, 0),
            vae=get_value_at_index(vae, 0)
        )
        NODE_CLASS_MAPPINGS["SaveImage"]().save_images(
            filename_prefix="ComfyUI",
            images=get_value_at_index(decoded, 0)
        )

    # Clean up
    os.remove(img_with_mask_path)
    print("Temporary mask file removed.")

if __name__ == "__main__":
    main()
