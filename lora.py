'''
 █████  ██      ██    ██  █████  ██████   ██████
██   ██ ██      ██    ██ ██   ██ ██   ██ ██    ██
███████ ██      ██    ██ ███████ ██████  ██    ██
██   ██ ██       ██  ██  ██   ██ ██   ██ ██    ██
██   ██ ███████   ████   ██   ██ ██   ██  ██████

# © 2025 Argos Lab. All rights reserved.
# Author: Troveski
# License: MIT
'''

import os
import random
import sys
import tempfile
import shutil
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
    # print(f"Temporarily changed sys.argv to: {sys.argv}") # Keep this commented unless debugging
    try:
        yield
    finally:
        sys.argv = original_argv
        # print(f"Restored sys.argv to: {sys.argv}") # Keep this commented unless debugging

def initialize_comfyui():
    """
    Safely initialize ComfyUI components after main args are parsed.
    Uses temp_sys_argv to prevent ComfyUI's internal parser conflicts.
    Also creates a dummy PromptServer instance required by some nodes.
    """
    # Use the context manager here to protect imports/initialization
    with temp_sys_argv([sys.argv[0]]): # Pass just the script name
        print("Attempting ComfyUI imports and initialization...")
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
            # print("Imported load_extra_path_config from main.py") # Commented
        except ImportError:
            # print("Warning: Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead.") # Commented
            try:
                from extra_config import load_extra_path_config # Import directly now utils is on path
                # print("Imported load_extra_path_config from utils.extra_config") # Commented
            except ImportError:
                print("ERROR: Could not import load_extra_path_config from either main.py or utils.extra_config!")
                raise

        extra_model_paths = find_path("extra_model_paths.yaml")
        if extra_model_paths:
            load_extra_path_config(extra_model_paths)
            # print(f"Loaded extra_model_paths from {extra_model_paths}") # Commented
        else:
            # print("No extra_model_paths.yaml found.") # Commented
             pass


        # Import these necessary modules and setup minimal server/execution environment
        # print("Initializing ComfyUI server/execution components for node loading...") # Commented
        import asyncio, execution, server
        from nodes import init_extra_nodes

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Create a dummy PromptServer instance
        try:
            if not hasattr(server.PromptServer, "instance") or server.PromptServer.instance is None:
                 # print("Creating dummy PromptServer instance...") # Commented
                 server.PromptServer.instance = server.PromptServer(loop=loop)
                 # Dummy queue might also be needed for some nodes
                 # execution.PromptQueue(server.PromptServer.instance)

        except Exception as e:
            print(f"Error creating dummy PromptServer instance: {e}")
            print("Some custom nodes that rely on PromptServer.instance might fail to load.")
            import traceback
            traceback.print_exc()
            # Don't re-raise, attempt to proceed with node loading

        # Initializing custom nodes
        # print("Calling init_extra_nodes()...") # Commented
        try:
             init_extra_nodes() # Load custom nodes into NODE_CLASS_MAPPINGS
             # print("init_extra_nodes() completed.") # Commented
        except Exception as e:
             print(f"Warning: Error during init_extra_nodes: {e}")
             print("This might mean some custom nodes failed to load.")
             import traceback
             traceback.print_exc()

        # print("ComfyUI core components initialization attempt finished.") # Commented
    # sys.argv is restored automatically by the context manager

# ────────────────────────────────

def augment_single_image(masked_image_path, original_label_path, output_image_dir, output_label_dir, index,
                         prompt, negative_prompt, lora_name, lora_strength_model,
                         lora_strength_clip, guidance, steps):
    """
    Process a single pre-masked image using the LoRA-based inpainting pipeline.
    Assumes initialize_comfyui() has been called.
    """
    # Import nodes mapping here AFTER initialization is complete
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        print("Error: Could not import NODE_CLASS_MAPPINGS. ComfyUI initialization likely failed.")
        raise

    # Verify the required nodes are loaded (optional, but good practice)
    required_nodes = ["UNETLoader", "DualCLIPLoader", "LoraLoader", "CLIPTextEncode", "VAELoader",
                      "KSamplerSelect", "RandomNoise", "LoadAndResizeImage", "ImpactGaussianBlurMask",
                      "FluxGuidance", "InpaintModelConditioning", "DifferentialDiffusion", "BasicGuider",
                      "BasicScheduler", "SamplerCustomAdvanced", "VAEDecode"]

    for node_name in required_nodes:
        if node_name not in NODE_CLASS_MAPPINGS:
            print(f"FATAL ERROR: Required node '{node_name}' is not loaded in NODE_CLASS_MAPPINGS.")
            raise RuntimeError(f"Required ComfyUI node '{node_name}' not found. Check custom node installations and initialization.")

    # Prepare output filenames (using original base name + augmentation index)
    original_basename = os.path.splitext(os.path.basename(original_label_path))[0] # Get base name from label file
    output_prefix = f"{original_basename}_{index}"
    output_image_path = os.path.join(output_image_dir, f"{output_prefix}.jpg")
    output_label_path = os.path.join(output_label_dir, f"{output_prefix}.txt")


    with torch.inference_mode():
        # -------------------------------------------------------------
        # LoRA-based Pipeline Logic
        # -------------------------------------------------------------

        # Load base models
        # Note: Loading models inside the loop for each image can be inefficient.
        # For performance, consider moving these outside the image loop if memory allows.
        # However, keeping them here ensures clean state per image processing.
        unet = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet("flux1-dev.safetensors", weight_dtype="fp8_e5m2")
        dualclip = NODE_CLASS_MAPPINGS["DualCLIPLoader"]().load_clip(
            clip_name1="t5xxl_fp16.safetensors", # ## FIX ## Corrected typo here
            clip_name2="clip_l.safetensors",   # ## FIX ## Corrected typo here
            type="flux",
            device="default"
        )

        # Load LoRA and apply it
        loraloader = NODE_CLASS_MAPPINGS["LoraLoader"]()
        lora_loaded = loraloader.load_lora(
            lora_name=lora_name,
            strength_model=lora_strength_model,
            strength_clip=lora_strength_clip,
            model=get_value_at_index(unet, 0),
            clip=get_value_at_index(dualclip, 0),
        )

        # Encode prompts
        text_enc = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        prompt_pos = text_enc.encode(text=prompt, clip=get_value_at_index(lora_loaded, 1))
        prompt_neg = text_enc.encode(text=negative_prompt, clip=get_value_at_index(lora_loaded, 1))

        # Load other components
        vae = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("flux_vae.safetensors")
        sampler_name = NODE_CLASS_MAPPINGS["KSamplerSelect"]().get_sampler("euler_ancestral")
        noise = NODE_CLASS_MAPPINGS["RandomNoise"]().get_noise(noise_seed=random.randint(1, 2 ** 64))

        # Load the PRE-MASKED input image
        load_img = NODE_CLASS_MAPPINGS["LoadAndResizeImage"]()
        img_and_mask = load_img.load_image(
            image=masked_image_path, # Use the path to the temp masked image
            resize=True, width=640, height=640, # Ensure this size is appropriate for your model/LoRA
            repeat=1,
            keep_proportion=True,
            divisible_by=1,
            mask_channel="alpha", # Use alpha channel as the mask
            background_color=""
        )

        # Blur mask
        mask_blur = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]().doit(
            kernel_size=5, sigma=5, mask=get_value_at_index(img_and_mask, 1) # Use the mask loaded with the image
        )

        # Set up guidance and inpainting conditioning
        guidance_node = NODE_CLASS_MAPPINGS["FluxGuidance"]().append(
            guidance=guidance, conditioning=get_value_at_index(prompt_pos, 0)
        )
        inpaint_cond = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]().encode(
            noise_mask=True, # This injects noise only where the mask is
            positive=get_value_at_index(guidance_node, 0),
            negative=get_value_at_index(prompt_neg, 0),
            vae=get_value_at_index(vae, 0),
            pixels=get_value_at_index(img_and_mask, 0), # The image pixels
            mask=get_value_at_index(mask_blur, 0)      # The blurred mask
        )

        # Diffusion and generation
        diff_model = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]().apply(model=get_value_at_index(lora_loaded, 0))
        guider = NODE_CLASS_MAPPINGS["BasicGuider"]().get_guider(
            model=get_value_at_index(diff_model, 0), conditioning=get_value_at_index(inpaint_cond, 0)
        )
        sigmas = NODE_CLASS_MAPPINGS["BasicScheduler"]().get_sigmas(
            scheduler="karras", steps=steps, denoise=1, model=get_value_at_index(lora_loaded, 0)
        )
        samples = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]().sample(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guider, 0),
            sampler=get_value_at_index(sampler_name, 0),
            sigmas=get_value_at_index(sigmas, 0),
            latent_image=get_value_at_index(inpaint_cond, 2)
        )

        # Decode and save image
        decoded = NODE_CLASS_MAPPINGS["VAEDecode"]().decode(
            samples=get_value_at_index(samples, 0), vae=get_value_at_index(vae, 0)
        )
        tensor = get_value_at_index(decoded, 0)[0]
        i = 255. * tensor.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        img.save(output_image_path)

        # Copy the original label file to match the augmented image
        shutil.copy(original_label_path, output_label_path)

    # Note: The temp masked image is deleted in process_dataset AFTER the inner loop
    return output_image_path

def process_dataset(args):
    """Process an entire dataset with YOLOv7 structure using provided arguments."""
    # Setup output directory structure
    output_images_dir = os.path.join(args.output, "images")
    output_labels_dir = os.path.join(args.output, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print("--- Initializing ComfyUI Environment (including custom nodes) ---")
    initialize_comfyui()
    print("--- ComfyUI Environment Initialized ---")

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

    # Process each image
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_images_dir, image_file)
        label_path = os.path.join(input_labels_dir, f"{base_name}.txt")

        # ## NEW ## Create the masked image and check if the target class was found
        # Create mask here, BEFORE the augmentation loop
        masked_img = create_alpha_mask_from_yolo(image_path, label_path, args.target_class)

        # ## NEW ## Check the flag attached by create_alpha_mask_from_yolo
        if not hasattr(masked_img, 'found_target_class') or not masked_img.found_target_class:
             print(f"Info: Target class {args.target_class} not found in {image_file}. Skipping augmentation.")
             # Copy original if requested, even if skipping augmentation
             if not args.no_originals:
                shutil.copy(image_path, os.path.join(output_images_dir, image_file))
                # Copy label file only if it exists
                if os.path.exists(label_path):
                    shutil.copy(label_path, os.path.join(output_labels_dir, f"{base_name}.txt"))
                print(f"Copied original: {image_file}")
             continue # Skip to the next image file

        # If target class was found, proceed with augmentation
        print(f"Target class {args.target_class} found in {image_file}. Proceeding with augmentation ({args.k} times).")

        # Save the masked image to a temporary file to be loaded by ComfyUI nodes
        tmp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir="/tmp")
        masked_img.save(tmp_file.name)
        img_with_mask_path = tmp_file.name # Path to the temporary masked image

        # Copy original image and label if requested (now happens only if target class is found)
        if not args.no_originals:
             # Original was already copied above before the continue, but if we move the continue,
             # this needs to be here. Let's consolidate.
             # Let's copy the original *only* if augmentation happens AND not args.no_originals
             # It's simpler to copy it always if not no_originals and the label exists, then skip aug.
             # The current logic (copying before the continue) is fine. No change needed here.
             pass


        # Create augmentations using the masked image file
        for i in range(args.k):
            try:
                img_path = augment_single_image(
                    img_with_mask_path, # Pass the temporary masked image path
                    label_path,       # Pass original label path for copying
                    output_images_dir,
                    output_labels_dir,
                    i,
                    # target_class is handled by the mask generation
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt,
                    lora_name=args.lora_name,
                    lora_strength_model=args.lora_strength_model,
                    lora_strength_clip=args.lora_strength_clip,
                    guidance=args.guidance,
                    steps=args.steps
                )
                print(f"Successfully created augmentation {i}: {os.path.basename(img_path)}")
            except Exception as e:
                print(f"ERROR processing {image_file} (augmentation {i}): {e}")
                import traceback
                traceback.print_exc() # Print detailed error

        # Clean up the temporary masked image file after processing this image
        try:
            os.remove(img_with_mask_path)
            # print(f"Cleaned up temporary file: {img_with_mask_path}") # Commented
        except OSError as e:
            print(f"Warning: Could not remove temporary file {img_with_mask_path}: {e}")


def main():
    # This parser runs FIRST and handles our custom arguments.
    parser = argparse.ArgumentParser(description="Batch augment a YOLO dataset using a LoRA-based ComfyUI inpainting pipeline.")
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
    args = parser.parse_args() # Our arguments are parsed successfully here.

    # Now call the function that includes the ComfyUI initialization
    process_dataset(args)

    print(f"\nAugmentation complete. Dataset saved to {args.output}")

if __name__ == "__main__":
    main()