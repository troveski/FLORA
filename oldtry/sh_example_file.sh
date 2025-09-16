#!/bin/bash

# Ativar o ambiente conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate comfyui

# Rodar todos os comandos em sequência - ROBOMASTER

python lora.py \
    --input underwater_train_200 \
    --output output_lora/underwater_class_0 \
    --k 4 \
    --target-class 0 \
    --prompt "underwater-echinus, a underwater image" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_underwater_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15
    
python lora.py \
    --input underwater_train_200 \
    --output output_lora/underwater_class_1 \
    --k 7 \
    --target-class 1 \
    --prompt "underwater-holothurian, a underwater image" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_underwater_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15
    
python lora.py \
    --input underwater_train_200 \
    --output output_lora/underwater_class_2 \
    --k 13 \
    --target-class 2 \
    --prompt "underwater-scallop, a underwater image" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_underwater_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15
    
python lora.py \
    --input underwater_train_200 \
    --output output_lora/underwater_class_3 \
    --k 6 \
    --target-class 3 \
    --prompt "underwater-starfish, a underwater image" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_underwater_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15
    
python lora.py \
    --input underwater_train_200 \
    --output output_lora/underwater_class_4 \
    --k 25 \
    --target-class 4 \
    --prompt "underwater-waterweeds, a underwater image" \
    --negative-prompt "blurry, low quality, cartoon, watermark, signature" \
    --lora-name "lora_underwater_2.0.safetensors" \
    --lora-strength-model 1.25 \
    --guidance 6.5 \
    --steps 15
    


