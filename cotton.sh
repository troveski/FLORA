#!/bin/bash

# Ativar o ambiente conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comfyui

# Rodar todos os comandos em sequência

python lora.py \
    --input train_200_cotton \
    --output output_lora/cotton_class_0 \
    --k 10 \
    --target-class 0 \
    --prompt "arboreum-cotton, a cotton plant photo" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_cotton_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15
    
    
python lora.py \
    --input train_200_cotton \
    --output output_lora/cotton_class_1 \
    --k 12 \
    --target-class 1 \
    --prompt "barbadense-cotton, a cotton plant photo" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_cotton_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 16
    
python lora.py \
    --input train_200_cotton \
    --output output_lora/cotton_class_2 \
    --k 11 \
    --target-class 2 \
    --prompt "herbaceum-cotton, a cotton plant photo" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_cotton_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15

python lora.py \
    --input train_200_cotton \
    --output output_lora/cotton_class_3 \
    --k 8 \
    --target-class 3 \
    --prompt "hirsitum-cotton, a cotton plant photo" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_cotton_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15
    





 

