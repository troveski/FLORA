'''
 █████  ██      ██    ██  █████  ██████   ██████
██   ██ ██      ██    ██ ██   ██ ██   ██ ██    ██
███████ ██      ██    ██ ███████ ██████  ██    ██
██   ██ ██       ██  ██  ██   ██ ██   ██ ██    ██
██   ██ ███████   ████   ██   ██ ██   ██  ██████

# © 2025 Argos Lab. All rights reserved.
# Author: Troveski
# License: MIT
# OPTIMIZED VERSION - Models loaded once for better performance
'''

import os
import random
import sys
import tempfile
import shutil
import time
from typing import Sequence, Mapping, Any, Union
import argparse
import contextlib

# These imports are safe as they don't trigger ComfyUI's parser directly.
import torch
from PIL import Image
import numpy as np

# ───────────────────────────────
# YOLO utilities
def create_alpha_mask_from_yolo(image_path: str, label_path: str, target_class: int) -> Image.Image:
    """
    Generate RGBA image with alpha=0 in regions matching specified class from YOLO labels.
    Returns the RGBA image.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    px = img.load()
    found_target_class = False # Flag to check if target class was found

    if not os.path.exists(label_path):
         print(f"Warning: Label file not found for {os.path.basename(image_path)} at {label_path}")
         return img # Return original image with full alpha if no label

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            if int(cls) != target_class:
                continue

            found_target_class = True # Target class found, set flag

            xmin = int((xc - bw / 2) * w)
            ymin = int((yc - bh / 2) * h)
            xmax = int((xc + bw / 2) * w)
            ymax = int((yc + bh / 2) * h)

            # Clamp coordinates to image boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w, xmax)
            ymax = min(h, ymax)

            # Apply transparency to the bounding box region
            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    r, g, b, _ = px[x, y]
                    px[x, y] = (r, g, b, 0)  # Set transparency

    img.found_target_class = found_target_class # Attach the flag to the image object
    return img
# ───────────────────────────────

# ───── ComfyUI utilities (Definitions only) ─────
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

@contextlib.contextmanager
def temp_sys_argv(args):
    """Temporarily replaces sys.argv with the given args."""
    original_argv = sys.argv
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = original_argv

def initialize_comfyui():
    """
    Safely initialize ComfyUI components after main args are parsed.
    Uses temp_sys_argv to prevent ComfyUI's internal parser conflicts.
    Also creates a dummy PromptServer instance required by some nodes.
    """
    init_start = time.time()
    print("🔄 Initializing ComfyUI environment...")
    
    # Use the context manager here to protect imports/initialization
    with temp_sys_argv([sys.argv[0]]): # Pass just the script name
        comfy_path = find_path("ComfyUI")
        if comfy_path and os.path.isdir(comfy_path):
            if comfy_path not in sys.path:
                 sys.path.append(comfy_path)
            comfy_utils_path = os.path.join(comfy_path, "utils")
            if os.path.isdir(comfy_utils_path) and comfy_utils_path not in sys.path:
                 sys.path.append(comfy_utils_path)

        # These imports can now happen safely
        try:
            from main import load_extra_path_config
        except ImportError:
            try:
                from extra_config import load_extra_path_config
            except ImportError:
                print("ERROR: Could not import load_extra_path_config from either main.py or utils.extra_config!")
                raise

        extra_model_paths = find_path("extra_model_paths.yaml")
        if extra_model_paths:
            load_extra_path_config(extra_model_paths)

        # Import these necessary modules and setup minimal server/execution environment
        import asyncio, execution, server
        from nodes import init_extra_nodes

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Create a dummy PromptServer instance
        try:
            if not hasattr(server.PromptServer, "instance") or server.PromptServer.instance is None:
                 server.PromptServer.instance = server.PromptServer(loop=loop)
        except Exception as e:
            print(f"Warning: Error creating dummy PromptServer instance: {e}")

        # Initializing custom nodes
        try:
             init_extra_nodes() # Load custom nodes into NODE_CLASS_MAPPINGS
        except Exception as e:
             print(f"Warning: Error during init_extra_nodes: {e}")
             import traceback
             traceback.print_exc()

    init_end = time.time()
    print(f"✅ ComfyUI environment initialized in {init_end - init_start:.2f}s")

def initialize_models(lora_name, lora_strength_model, lora_strength_clip):
    """Load all models once and return them for reuse."""
    model_start = time.time()
    print("🔄 Loading models (this may take a while)...")
    
    from nodes import NODE_CLASS_MAPPINGS
    
    # Verify required nodes are loaded
    required_nodes = ["UNETLoader", "DualCLIPLoader", "LoraLoader", "CLIPTextEncode", "VAELoader",
                      "KSamplerSelect", "RandomNoise", "LoadAndResizeImage", "ImpactGaussianBlurMask",
                      "FluxGuidance", "InpaintModelConditioning", "DifferentialDiffusion", "BasicGuider",
                      "BasicScheduler", "SamplerCustomAdvanced", "VAEDecode"]

    missing_nodes = [node for node in required_nodes if node not in NODE_CLASS_MAPPINGS]
    if missing_nodes:
        print(f"FATAL ERROR: Missing required nodes: {missing_nodes}")
        raise RuntimeError(f"Required ComfyUI nodes not found: {missing_nodes}")

    print("  📦 Loading UNET model...")
    unet_start = time.time()
    unet = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet("flux1-dev.safetensors", weight_dtype="fp8_e5m2")
    unet_end = time.time()
    print(f"  ✅ UNET loaded in {unet_end - unet_start:.2f}s")
    
    print("  📦 Loading CLIP models...")
    clip_start = time.time()
    dualclip = NODE_CLASS_MAPPINGS["DualCLIPLoader"]().load_clip(
        clip_name1="t5xxl_fp16.safetensors",
        clip_name2="clip_l.safetensors",
        type="flux",
        device="default"
    )
    clip_end = time.time()
    print(f"  ✅ CLIP models loaded in {clip_end - clip_start:.2f}s")
    
    print("  📦 Loading LoRA...")
    lora_start = time.time()
    loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
    lora_loaded = loraloader.load_lora(
        lora_name=lora_name,
        strength_model=lora_strength_model,
        strength_clip=lora_strength_clip,
        model=get_value_at_index(unet, 0),
        clip=get_value_at_index(dualclip, 0),
    )
    lora_end = time.time()
    print(f"  ✅ LoRA loaded in {lora_end - lora_start:.2f}s")
    
    print("  📦 Loading VAE...")
    vae_start = time.time()
    vae = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("flux_vae.safetensors")
    vae_end = time.time()
    print(f"  ✅ VAE loaded in {vae_end - vae_start:.2f}s")
    
    print("  📦 Initializing sampler...")
    sampler_start = time.time()
    sampler_name = NODE_CLASS_MAPPINGS["KSamplerSelect"]().get_sampler("euler_ancestral")
    sampler_end = time.time()
    print(f"  ✅ Sampler initialized in {sampler_end - sampler_start:.2f}s")
    
    model_end = time.time()
    print(f"🚀 All models loaded successfully in {model_end - model_start:.2f}s")
    
    return {
        'model': get_value_at_index(lora_loaded, 0),
        'clip': get_value_at_index(lora_loaded, 1),
        'vae': get_value_at_index(vae, 0),
        'sampler': get_value_at_index(sampler_name, 0)
    }

def augment_single_image_optimized(masked_image_path, original_label_path, output_image_dir, output_label_dir, 
                                 index, prompt, negative_prompt, guidance, steps, models):
    """
    Optimized version that reuses pre-loaded models.
    """
    from nodes import NODE_CLASS_MAPPINGS
    
    # Prepare output filenames
    original_basename = os.path.splitext(os.path.basename(original_label_path))[0]
    output_prefix = f"{original_basename}_{index}"
    output_image_path = os.path.join(output_image_dir, f"{output_prefix}.jpg")
    output_label_path = os.path.join(output_label_dir, f"{output_prefix}.txt")

    with torch.inference_mode():
        # Use pre-loaded models instead of loading each time
        text_enc = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        prompt_pos = text_enc.encode(text=prompt, clip=models['clip'])
        prompt_neg = text_enc.encode(text=negative_prompt, clip=models['clip'])
        
        # Generate new noise for each image
        noise = NODE_CLASS_MAPPINGS["RandomNoise"]().get_noise(noise_seed=random.randint(1, 2 ** 64))
        
        # Load the image (this is per-image, unavoidable)
        load_img = NODE_CLASS_MAPPINGS["LoadAndResizeImage"]()
        img_and_mask = load_img.load_image(
            image=masked_image_path,
            resize=True, width=640, height=640,
            repeat=1,
            keep_proportion=True,
            divisible_by=1,
            mask_channel="alpha",
            background_color=""
        )
        
        # Blur mask
        mask_blur = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]().doit(
            kernel_size=5, sigma=5, mask=get_value_at_index(img_and_mask, 1)
        )
        
        # Set up guidance and inpainting conditioning
        guidance_node = NODE_CLASS_MAPPINGS["FluxGuidance"]().append(
            guidance=guidance, conditioning=get_value_at_index(prompt_pos, 0)
        )
        
        inpaint_cond = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]().encode(
            noise_mask=True,
            positive=get_value_at_index(guidance_node, 0),
            negative=get_value_at_index(prompt_neg, 0),
            vae=models['vae'],
            pixels=get_value_at_index(img_and_mask, 0),
            mask=get_value_at_index(mask_blur, 0)
        )
        
        # Diffusion and generation
        diff_model = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]().apply(model=models['model'])
        guider = NODE_CLASS_MAPPINGS["BasicGuider"]().get_guider(
            model=get_value_at_index(diff_model, 0), 
            conditioning=get_value_at_index(inpaint_cond, 0)
        )
        
        sigmas = NODE_CLASS_MAPPINGS["BasicScheduler"]().get_sigmas(
            scheduler="karras", steps=steps, denoise=1, model=models['model']
        )
        
        samples = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]().sample(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guider, 0),
            sampler=models['sampler'],
            sigmas=get_value_at_index(sigmas, 0),
            latent_image=get_value_at_index(inpaint_cond, 2)
        )
        
        # Decode and save image
        decoded = NODE_CLASS_MAPPINGS["VAEDecode"]().decode(
            samples=get_value_at_index(samples, 0), vae=models['vae']
        )
        
        tensor = get_value_at_index(decoded, 0)[0]
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(output_image_path)
        
        # Copy the original label file to match the augmented image
        shutil.copy(original_label_path, output_label_path)
    
    return output_image_path

def process_dataset(args):
    """Process an entire dataset with YOLOv7 structure using provided arguments."""
    total_start = time.time()
    print("🚀 Starting dataset processing...")
    
    # Setup output directory structure
    output_images_dir = os.path.join(args.output, "images")
    output_labels_dir = os.path.join(args.output, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Initialize ComfyUI
    initialize_comfyui()
    
    # Load models once
    models = initialize_models(args.lora_name, args.lora_strength_model, args.lora_strength_clip)

    # Get input directory structure
    input_images_dir = os.path.join(args.input, "images")
    input_labels_dir = os.path.join(args.input, "labels")

    if not os.path.isdir(input_images_dir):
        print(f"ERROR: Input image directory not found: {input_images_dir}")
        return
    if not os.path.isdir(input_labels_dir):
         print(f"ERROR: Input label directory not found: {input_labels_dir}")
         return

    # Get list of all images
    image_files = [f for f in os.listdir(input_images_dir)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in {input_images_dir}. Nothing to process.")
        return

    print(f"📊 Found {len(image_files)} images to process")
    print(f"🎯 Target class: {args.target_class}")
    print(f"📈 Augmentations per image: {args.k}")
    print("=" * 60)

    # Process each image
    processed_count = 0
    augmented_count = 0
    skipped_count = 0
    
    for idx, image_file in enumerate(image_files, 1):
        image_start = time.time()
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_images_dir, image_file)
        label_path = os.path.join(input_labels_dir, f"{base_name}.txt")

        print(f"\n[{idx}/{len(image_files)}] Processing: {image_file}")

        # Create the masked image and check if the target class was found
        mask_start = time.time()
        masked_img = create_alpha_mask_from_yolo(image_path, label_path, args.target_class)
        mask_end = time.time()

        # Check if target class was found
        if not hasattr(masked_img, 'found_target_class') or not masked_img.found_target_class:
             print(f"  ⚠️  Target class {args.target_class} not found - skipping augmentation")
             skipped_count += 1
             # Copy original if requested
             if not args.no_originals:
                shutil.copy(image_path, os.path.join(output_images_dir, image_file))
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(output_labels_dir, f"{base_name}.txt"))
                print(f"  📄 Copied original files")
             continue

        processed_count += 1
        print(f"  ✅ Target class found! Creating mask took {mask_end - mask_start:.2f}s")

        # Save the masked image to a temporary file
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
        masked_img.save(tmp_file.name)
        img_with_mask_path = tmp_file.name

        # Copy original image and label if requested
        if not args.no_originals:
            shutil.copy(image_path, os.path.join(output_images_dir, image_file))
            if os.path.exists(label_path):
                shutil.copy(label_path, os.path.join(output_labels_dir, f"{base_name}.txt"))

        # Create augmentations using the masked image file
        print(f"  🎨 Generating {args.k} augmentations...")
        for i in range(args.k):
            try:
                aug_start = time.time()
                img_path = augment_single_image_optimized(
                    img_with_mask_path,
                    label_path,
                    output_images_dir,
                    output_labels_dir,
                    i,
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    guidance=args.guidance,
                    steps=args.steps,
                    models=models
                )
                aug_end = time.time()
                augmented_count += 1
                print(f"    ✅ Aug {i+1}/{args.k} completed in {aug_end - aug_start:.2f}s: {os.path.basename(img_path)}")
                
                # Clear cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"    ❌ ERROR in augmentation {i}: {e}")
                import traceback
                traceback.print_exc()

        # Clean up the temporary masked image file
        try:
            os.remove(img_with_mask_path)
        except OSError as e:
            print(f"    ⚠️  Warning: Could not remove temp file: {e}")

        image_end = time.time()
        print(f"  ⏱️  Total time for {image_file}: {image_end - image_start:.2f}s")

    total_end = time.time()
    print("\n" + "=" * 60)
    print("📊 PROCESSING SUMMARY")
    print("=" * 60)
    print(f"📁 Total images found: {len(image_files)}")
    print(f"✅ Images with target class: {processed_count}")
    print(f"⚠️  Images skipped (no target class): {skipped_count}")
    print(f"🎨 Total augmentations created: {augmented_count}")
    print(f"⏱️  Total processing time: {total_end - total_start:.2f}s")
    if processed_count > 0:
        print(f"📈 Average time per processed image: {(total_end - total_start)/processed_count:.2f}s")
    print(f"💾 Output saved to: {args.output}")

def main():
    # This parser runs FIRST and handles our custom arguments.
    parser = argparse.ArgumentParser(description="OPTIMIZED: Batch augment a YOLO dataset using a LoRA-based ComfyUI inpainting pipeline.")
    parser.add_argument("--input", required=True, help="Input dataset directory (with images/ and labels/ subdirectories).")
    parser.add_argument("--output", default="dataset_augmented", help="Output directory for the augmented dataset.")
    parser.add_argument("--k", type=int, default=3, help="Number of augmentations per image that CONTAINS the target class.")
    parser.add_argument("--target-class", type=int, required=True, help="YOLO class ID to target for inpainting.")
    parser.add_argument("--no-originals", action="store_true", help="Don't include original images and labels in the output.")
    parser.add_argument("--prompt", type=str, required=True, help="Positive prompt for the generation.")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, cartoon, watermark, signature", help="Negative prompt.")
    parser.add_argument("--lora-name", type=str, required=True, help="Filename of the LoRA model (e.g., 'my_lora.safetensors').")
    parser.add_argument("--lora-strength-model", type=float, default=1.0, help="Strength of the LoRA applied to the UNET model.")
    parser.add_argument("--lora-strength-clip", type=float, default=1.0, help="Strength of the LoRA applied to the CLIP model.")
    parser.add_argument("--guidance", type=float, default=6.5, help="Guidance scale (CFG).")
    parser.add_argument("--steps", type=int, default=15, help="Number of sampling steps.")
    args = parser.parse_args()

    # Display configuration
    print("🔧 CONFIGURATION")
    print("=" * 40)
    print(f"📂 Input: {args.input}")
    print(f"📂 Output: {args.output}")
    print(f"🎯 Target class: {args.target_class}")
    print(f"📈 Augmentations per image: {args.k}")
    print(f"🎨 LoRA: {args.lora_name}")
    print(f"💪 LoRA strength (model): {args.lora_strength_model}")
    print(f"💪 LoRA strength (clip): {args.lora_strength_clip}")
    print(f"🎛️  Guidance: {args.guidance}")
    print(f"⏱️  Steps: {args.steps}")
    print(f"➕ Prompt: {args.prompt}")
    print(f"➖ Negative: {args.negative_prompt}")
    print(f"📄 Include originals: {not args.no_originals}")
    print("=" * 40)

    # Process the dataset
    process_dataset(args)

if __name__ == "__main__":
    main()
