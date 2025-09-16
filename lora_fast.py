#  █████  ██      ██   ██ █████  ██████    ██████
# ██   ██ ██      ██   ██ ██   ██ ██   ██ ██    ██
# ███████ ██      ██   ██ ███████ ██████  ██    ██
# ██   ██ ██         ██ ██  ██   ██ ██   ██ ██    ██
# ██   ██ ███████   ████   ██   ██ ██   ██  ██████

# © 2025 Argos Lab. All rights reserved.
# Author: Troveski
# License: MIT

import os
import random
import sys
import tempfile
import shutil
from typing import Sequence, Mapping, Any, Union
import argparse
import contextlib
import time
from tqdm import tqdm

# Imports do PyTorch, PIL, etc.
import torch
from PIL import Image
import numpy as np

# ───────────────────────────────
# Funções Utilitárias YOLO
# ───────────────────────────────
def create_alpha_mask_from_yolo(image_path: str, label_path: str, target_class: int) -> Image.Image:
    """
    Gera uma imagem RGBA com canal alfa transparente nas regiões da classe alvo.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    px = img.load()
    found_target_class = False

    if not os.path.exists(label_path):
        img.found_target_class = False
        return img

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = map(float, parts)
            if int(cls) != target_class:
                continue

            found_target_class = True
            xmin = int((xc - bw / 2) * w)
            ymin = int((yc - bh / 2) * h)
            xmax = int((xc + bw / 2) * w)
            ymax = int((yc + bh / 2) * h)

            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(w, xmax), min(h, ymax)

            for y in range(ymin, ymax):
                for x in range(xmin, xmax):
                    r, g, b, _ = px[x, y]
                    px[x, y] = (r, g, b, 0)

    img.found_target_class = found_target_class
    return img

# ───────────────────────────────
# Funções Utilitárias ComfyUI
# ───────────────────────────────
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
    original_argv = sys.argv
    sys.argv = args
    try:
        yield
    finally:
        sys.argv = original_argv

def initialize_comfyui():
    """Inicializa o ambiente e os nós customizados do ComfyUI de forma segura."""
    with temp_sys_argv([sys.argv[0]]):
        comfy_path = find_path("ComfyUI")
        if comfy_path and os.path.isdir(comfy_path) and comfy_path not in sys.path:
            sys.path.append(comfy_path)
        comfy_utils_path = os.path.join(comfy_path, "utils")
        if os.path.isdir(comfy_utils_path) and comfy_utils_path not in sys.path:
            sys.path.append(comfy_utils_path)

        try:
            from main import load_extra_path_config
        except ImportError:
            try:
                from extra_config import load_extra_path_config
            except ImportError:
                print("ERRO: Não foi possível importar load_extra_path_config!")
                raise

        extra_model_paths = find_path("extra_model_paths.yaml")
        if extra_model_paths:
            load_extra_path_config(extra_model_paths)

        import asyncio, server
        from nodes import init_extra_nodes

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if not hasattr(server.PromptServer, "instance") or server.PromptServer.instance is None:
            server.PromptServer.instance = server.PromptServer(loop=loop)

        init_extra_nodes()

# ───────────────────────────────
# Lógica Principal da Pipeline
# ───────────────────────────────
def initialize_pipeline(args):
    """
    Carrega os modelos e inicializa todos os nós da pipeline da ComfyUI uma única vez.
    Retorna um dicionário contendo todos os componentes prontos para uso.
    """
    from nodes import NODE_CLASS_MAPPINGS

    print("⏳ Inicializando pipeline completa (Modelos e Nós)... Isso pode levar um momento.")
    pipeline = {}

    # 1. Carregar Modelos Base
    pipeline['unet_base'] = NODE_CLASS_MAPPINGS["UNETLoader"]().load_unet("flux1-dev.safetensors", weight_dtype="fp8_e5m2")
    pipeline['clip_base'] = NODE_CLASS_MAPPINGS["DualCLIPLoader"]().load_clip(
        clip_name1="t5xxl_fp16.safetensors", clip_name2="clip_l.safetensors", type="flux"
    )
    pipeline['vae'] = NODE_CLASS_MAPPINGS["VAELoader"]().load_vae("flux_vae.safetensors")

    # 2. Carregar e Aplicar LoRA
    lora_loaded = NODE_CLASS_MAPPINGS["LoraLoader"]().load_lora(
        lora_name=args.lora_name,
        strength_model=args.lora_strength_model,
        strength_clip=args.lora_strength_clip,
        model=get_value_at_index(pipeline['unet_base'], 0),
        clip=get_value_at_index(pipeline['clip_base'], 0),
    )
    pipeline['lora_model'] = get_value_at_index(lora_loaded, 0)
    pipeline['lora_clip'] = get_value_at_index(lora_loaded, 1)
    
    # 3. Inicializar instâncias de todos os Nós
    pipeline['text_encoder_node'] = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
    pipeline['sampler_selector_node'] = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
    pipeline['noise_generator_node'] = NODE_CLASS_MAPPINGS["RandomNoise"]()
    pipeline['image_loader_node'] = NODE_CLASS_MAPPINGS["LoadAndResizeImage"]()
    pipeline['mask_blur_node'] = NODE_CLASS_MAPPINGS["ImpactGaussianBlurMask"]()
    pipeline['guidance_node'] = NODE_CLASS_MAPPINGS["FluxGuidance"]()
    pipeline['inpaint_conditioner_node'] = NODE_CLASS_MAPPINGS["InpaintModelConditioning"]()
    pipeline['diffusion_node'] = NODE_CLASS_MAPPINGS["DifferentialDiffusion"]()
    pipeline['guider_node'] = NODE_CLASS_MAPPINGS["BasicGuider"]()
    pipeline['scheduler_node'] = NODE_CLASS_MAPPINGS["BasicScheduler"]()
    pipeline['sampler_node'] = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
    pipeline['decoder_node'] = NODE_CLASS_MAPPINGS["VAEDecode"]()

    print("✅ Pipeline inicializada com sucesso!")
    return pipeline

def augment_single_image(masked_image_path, original_label_path, output_image_dir, output_label_dir, index,
                         prompt, negative_prompt, guidance, steps, pipeline):
    """
    Processa uma única imagem usando a pipeline pré-inicializada.
    """
    original_basename = os.path.splitext(os.path.basename(original_label_path))[0]
    output_prefix = f"{original_basename}_{index}"
    output_image_path = os.path.join(output_image_dir, f"{output_prefix}.jpg")
    output_label_path = os.path.join(output_label_dir, f"{output_prefix}.txt")

    with torch.inference_mode():
        prompt_pos = pipeline['text_encoder_node'].encode(text=prompt, clip=pipeline['lora_clip'])
        prompt_neg = pipeline['text_encoder_node'].encode(text=negative_prompt, clip=pipeline['lora_clip'])
        
        sampler_name = pipeline['sampler_selector_node'].get_sampler("euler_ancestral")
        noise = pipeline['noise_generator_node'].get_noise(noise_seed=random.randint(1, 2 ** 64))

        img_and_mask = pipeline['image_loader_node'].load_image(
            image=masked_image_path,
            resize=True,
            width=640,
            height=640,
            repeat=1,
            keep_proportion=True,
            divisible_by=1,
            mask_channel="alpha",
            background_color=""
        )
        
        mask_blur = pipeline['mask_blur_node'].doit(kernel_size=5, sigma=5, mask=get_value_at_index(img_and_mask, 1))
        guidance_obj = pipeline['guidance_node'].append(guidance=guidance, conditioning=get_value_at_index(prompt_pos, 0))
        inpaint_cond = pipeline['inpaint_conditioner_node'].encode(
            noise_mask=True,
            positive=get_value_at_index(guidance_obj, 0),
            negative=get_value_at_index(prompt_neg, 0),
            vae=get_value_at_index(pipeline['vae'], 0),
            pixels=get_value_at_index(img_and_mask, 0),
            mask=get_value_at_index(mask_blur, 0)
        )
        diff_model = pipeline['diffusion_node'].apply(model=pipeline['lora_model'])
        guider = pipeline['guider_node'].get_guider(
            model=get_value_at_index(diff_model, 0), conditioning=get_value_at_index(inpaint_cond, 0)
        )
        sigmas = pipeline['scheduler_node'].get_sigmas(
            scheduler="karras", steps=steps, denoise=1, model=pipeline['lora_model']
        )
        samples = pipeline['sampler_node'].sample(
            noise=get_value_at_index(noise, 0),
            guider=get_value_at_index(guider, 0),
            sampler=get_value_at_index(sampler_name, 0),
            sigmas=get_value_at_index(sigmas, 0),
            latent_image=get_value_at_index(inpaint_cond, 2)
        )
        decoded = pipeline['decoder_node'].decode(
            samples=get_value_at_index(samples, 0), vae=get_value_at_index(pipeline['vae'], 0)
        )

        tensor = get_value_at_index(decoded, 0)[0]
        img = Image.fromarray(np.clip(255. * tensor.cpu().numpy(), 0, 255).astype(np.uint8))
        img.save(output_image_path)

        if os.path.exists(original_label_path):
            shutil.copy(original_label_path, output_label_path)

def process_dataset(args):
    """Orquestra todo o processo de aumento de dataset."""
    output_images_dir = os.path.join(args.output, "images")
    output_labels_dir = os.path.join(args.output, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    print("--- Inicializando Ambiente ComfyUI ---")
    initialize_comfyui()
    
    pipeline = initialize_pipeline(args)

    input_images_dir = os.path.join(args.input, "images")
    input_labels_dir = os.path.join(args.input, "labels")

    all_image_files = [f for f in os.listdir(input_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # --- Lógica de Estimativa de Tempo ---
    print("🔍 Verificando imagens que contêm a classe alvo para estimar o trabalho...")
    image_files_to_process = [
        img_file for img_file in all_image_files
        if create_alpha_mask_from_yolo(
            os.path.join(input_images_dir, img_file),
            os.path.join(input_labels_dir, f"{os.path.splitext(img_file)[0]}.txt"),
            args.target_class
        ).found_target_class
    ]

    if not image_files_to_process:
        print("Nenhuma imagem com a classe alvo foi encontrada. O script será encerrado.")
        return

    total_augmentations = len(image_files_to_process) * args.k
    print(f"Encontradas {len(image_files_to_process)} imagens com a classe alvo. Serão geradas {total_augmentations} novas imagens.")
    
    # Medir tempo com uma imagem de amostra
    print("⏱️  Processando uma imagem de amostra para estimar o tempo total...")
    first_image_file = image_files_to_process[0]
    first_image_path = os.path.join(input_images_dir, first_image_file)
    first_label_path = os.path.join(input_labels_dir, f"{os.path.splitext(first_image_file)[0]}.txt")
    
    start_time_sample = time.time()
    with tempfile.TemporaryDirectory() as temp_dir:
        masked_img = create_alpha_mask_from_yolo(first_image_path, first_label_path, args.target_class)
        temp_mask_path = os.path.join(temp_dir, "temp_mask.png")
        masked_img.save(temp_mask_path)
        # Gera a imagem de teste em um diretório temporário para não poluir a saída final
        augment_single_image(
            temp_mask_path, first_label_path, temp_dir, temp_dir, "estimativa",
            args.prompt, args.negative_prompt, args.guidance, args.steps, pipeline
        )
    end_time_sample = time.time()

    time_per_augmentation = end_time_sample - start_time_sample
    total_estimated_seconds = time_per_augmentation * total_augmentations
    print(f"Tempo por aumento: {time_per_augmentation:.2f} segundos.")
    print(f"Tempo total estimado (aproximado): {time.strftime('%H horas, %M minutos e %S segundos', time.gmtime(total_estimated_seconds))}")
    
    # --- Processamento em Lote ---
    if not args.no_originals:
        print("Copiando arquivos originais para o diretório de saída...")
        for img_file in tqdm(all_image_files, desc="Copiando originais"):
            shutil.copy(os.path.join(input_images_dir, img_file), os.path.join(output_images_dir, img_file))
            label_p = os.path.join(input_labels_dir, f"{os.path.splitext(img_file)[0]}.txt")
            if os.path.exists(label_p):
                shutil.copy(label_p, os.path.join(output_labels_dir, f"{os.path.splitext(img_file)[0]}.txt"))

    print(f"Iniciando a geração de {total_augmentations} aumentos...")
    with tqdm(total=total_augmentations, desc="Gerando Aumentos") as pbar:
        for image_file in image_files_to_process:
            image_path = os.path.join(input_images_dir, image_file)
            label_path = os.path.join(input_labels_dir, f"{os.path.splitext(image_file)[0]}.txt")
            
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp_file:
                masked_img = create_alpha_mask_from_yolo(image_path, label_path, args.target_class)
                masked_img.save(tmp_file.name)
                
                for i in range(args.k):
                    try:
                        augment_single_image(
                            tmp_file.name, label_path, output_images_dir, output_labels_dir, i,
                            args.prompt, args.negative_prompt, args.guidance, args.steps, pipeline
                        )
                    except Exception as e:
                        print(f"\nERRO ao processar {image_file} (aumento {i}): {e}")
                    pbar.update(1)

# ───────────────────────────────
# Ponto de Entrada do Script
# ───────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Aumenta um dataset YOLO usando uma pipeline de inpainting com LoRA via ComfyUI.")
    parser.add_argument("--input", required=True, help="Diretório do dataset de entrada (deve conter subdiretórios images/ e labels/).")
    parser.add_argument("--output", default="dataset_augmented", help="Diretório de saída para o dataset aumentado.")
    parser.add_argument("--k", type=int, default=3, help="Número de aumentos por imagem que contém a classe alvo.")
    parser.add_argument("--target-class", type=int, required=True, help="ID da classe YOLO a ser usada para inpainting.")
    parser.add_argument("--no-originals", action="store_true", help="Não incluir as imagens e labels originais na saída.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt positivo para a geração.")
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, cartoon, watermark, signature", help="Prompt negativo.")
    parser.add_argument("--lora-name", type=str, required=True, help="Nome do arquivo do modelo LoRA (ex: 'meu_lora.safetensors').")
    parser.add_argument("--lora-strength-model", type=float, default=1.0, help="Força do LoRA aplicada ao modelo UNET.")
    parser.add_argument("--lora-strength-clip", type=float, default=1.0, help="Força do LoRA aplicada ao modelo CLIP.")
    parser.add_argument("--guidance", type=float, default=6.5, help="Escala de orientação (CFG).")
    parser.add_argument("--steps", type=int, default=15, help="Número de passos de amostragem (steps).")
    args = parser.parse_args()

    process_dataset(args)

    print(f"\n✨ Processo de aumento concluído. Dataset salvo em: {args.output}")

if __name__ == "__main__":
    main()
