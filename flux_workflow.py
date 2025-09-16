import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from PIL import Image

#### CERTO EM PAI
########################### FUCNTIONS ###############################

def create_alpha_mask_from_yolo(image_path, label_path, output_path, target_class=9):
    # Carrega a imagem e garante que ela tem canal alpha
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    pixels = img.load()

    # Lê o arquivo de labels YOLO
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, x_center, y_center, box_width, box_height = map(float, parts)
            if int(cls) != target_class:
                continue

            # Converte coordenadas YOLO para pixels
            xmin = int((x_center - box_width / 2) * w)
            ymin = int((y_center - box_height / 2) * h)
            xmax = int((x_center + box_width / 2) * w)
            ymax = int((y_center + box_height / 2) * h)

            # Aplica transparência no canal alpha para a região do carro
            for y in range(max(0, ymin), min(h, ymax)):
                for x in range(max(0, xmin), min(w, xmax)):
                    r, g, b, a = pixels[x, y]
                    pixels[x, y] = (r, g, b, 0)  # alpha = 0 → transparente

    # Salva a imagem modificada
    img.save(output_path)
    print(f"✅ Imagem com máscara alpha salva em: {output_path}")

##########################################################




######################### CONFIG #########################

image_normal="/home/ai4ar/alvaro/ComfyUI/car_test/car1.jpg"
label= "/home/ai4ar/alvaro/ComfyUI/car_test/car1.txt"

# Cria a imagem com canal alpha usando os labels YOLO
create_alpha_mask_from_yolo(
    image_path=image_normal,
    label_path=label,
    output_path="/home/ai4ar/alvaro/ComfyUI/car_test/car1_with_mask.png"
)

# Atualiza o caminho da imagem para o inpaint
img_with_mask = "/home/ai4ar/alvaro/ComfyUI/car_test/car1_with_mask.png"



##########################################################













def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_11 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp16.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text="A car", clip=get_value_at_index(dualcliploader_11, 0)
        )

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_10 = vaeloader.load_vae(
            vae_name="diffusion_pytorch_model.safetensors"
        )

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_12 = unetloader.load_unet(
            unet_name="flux1-dev.safetensors", weight_dtype="fp8_e5m2"
        )

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_16 = ksamplerselect.get_sampler(sampler_name="euler")

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
        randomnoise_25 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

        loadandresizeimage = NODE_CLASS_MAPPINGS["LoadAndResizeImage"]()
        loadandresizeimage_35 = loadandresizeimage.load_image(
            image= img_with_mask,
            resize=True,
            width=640,
            height=640,
            repeat=1,
            keep_proportion=True,
            divisible_by=1,
            mask_channel="alpha",
            background_color="",
        )

        fluxguidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        fluxguidance_39 = fluxguidance.append(
            guidance=3.5, conditioning=get_value_at_index(cliptextencode_6, 0)
        )

        cliptextencode_40 = cliptextencode.encode(
            text="", clip=get_value_at_index(dualcliploader_11, 0)
        )

        impactgaussianblurmask = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]()
        impactgaussianblurmask_48 = impactgaussianblurmask.doit(
            kernel_size=10, sigma=10, mask=get_value_at_index(loadandresizeimage_35, 1)
        )

        inpaintmodelconditioning = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
        inpaintmodelconditioning_37 = inpaintmodelconditioning.encode(
            noise_mask=True,
            positive=get_value_at_index(fluxguidance_39, 0),
            negative=get_value_at_index(cliptextencode_40, 0),
            vae=get_value_at_index(vaeloader_10, 0),
            pixels=get_value_at_index(loadandresizeimage_35, 0),
            mask=get_value_at_index(impactgaussianblurmask_48, 0),
        )

        differentialdiffusion = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            differentialdiffusion_49 = differentialdiffusion.apply(
                model=get_value_at_index(unetloader_12, 0)
            )

            basicguider_22 = basicguider.get_guider(
                model=get_value_at_index(differentialdiffusion_49, 0),
                conditioning=get_value_at_index(inpaintmodelconditioning_37, 0),
            )

            basicscheduler_17 = basicscheduler.get_sigmas(
                scheduler="simple",
                steps=15,
                denoise=1,
                model=get_value_at_index(unetloader_12, 0),
            )

            samplercustomadvanced_13 = samplercustomadvanced.sample(
                noise=get_value_at_index(randomnoise_25, 0),
                guider=get_value_at_index(basicguider_22, 0),
                sampler=get_value_at_index(ksamplerselect_16, 0),
                sigmas=get_value_at_index(basicscheduler_17, 0),
                latent_image=get_value_at_index(inpaintmodelconditioning_37, 2),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(samplercustomadvanced_13, 0),
                vae=get_value_at_index(vaeloader_10, 0),
            )

            saveimage_9 = saveimage.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0)
            )


if __name__ == "__main__":
    main()
