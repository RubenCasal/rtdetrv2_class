#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="rtdetr_env"
PY_VER="3.12.11"

# 0) Preparar conda en shell no interactiva
eval "$(conda shell.bash hook)"

# 1) Crear y activar entorno
conda create -y -n "$ENV_NAME" "python=${PY_VER}" pip
conda activate "$ENV_NAME"

# 2) PyTorch + CUDA 12.8 (exacto)
pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.8.0+cu128" \
  "torchvision==0.23.0+cu128"

# 3) Pila HF/Datasets + visión + entrenamiento (pines exactos)
pip install --no-cache-dir \
  "datasets==2.20.0" \
  "huggingface-hub==0.35.3" \
  "pyarrow==18.1.0" \
  "fsspec==2024.5.0" \
  "xxhash==3.5.0" \
  "aiohttp==3.12.15" \
  "numpy==2.0.2" \
  "pillow==11.3.0" \
  "opencv-python==4.12.0.88" \
  "opencv-contrib-python==4.12.0.88" \
  "opencv-python-headless==4.12.0.88" \
  "albumentations==1.4.6" \
  "pycocotools==2.0.10" \
  "transformers==4.49.0" \
  "accelerate==1.10.1" \
  "torchmetrics==1.8.2" \
  "tensorboardX==2.6.4" \
  "matplotlib==3.10.6" \
  "timm==0.9.16"

# 4) Verificación rápida
python - <<'PY'
import importlib, sys, numpy as np
def v(m):
    try:
        M = importlib.import_module(m); return getattr(M, "__version__", "?")
    except Exception as e:
        return f"ERR({type(e).__name__})"

mods = ["torch","torchvision","datasets","huggingface_hub","pyarrow","fsspec","xxhash","aiohttp",
        "numpy","PIL","cv2","albumentations","pycocotools","transformers","accelerate","torchmetrics","timm"]
print("🐍", sys.version)
for m in mods: print(f"{m:<24}", v(m))

