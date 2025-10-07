# Training RT‑DETRv2

This section explains how to run the main training entrypoint for the RT‑DETRv2 model in this repository.

## Environment

Install all dependencies with the provided script. It creates a Conda environment with **Python 3.12.11** and pinned package versions.

```bash
chmod +x create_env.sh
./create_env.sh
conda activate rtdetr_env
```

## Data

`DATASET_PATH` must point to a dataset compatible with your `load_custom_dataset` implementation (COCO‑style boxes `[x, y, w, h]`, class indices, and `train/validation/test` splits).

## Configure

Open `main_train.py` (or your chosen entry file) and set:

* `MODE = "train"`
* `CHECKPOINT` → path to the pretrained weights or a local checkpoint
* `DATASET_PATH`, `OUTPUT_DIR`, `IMAGE_SIZE`, `TRAINING_EPOCHS`, `WARMUP_STEPS`, `LEARNING_RATE`, `BATCH_SIZE`

## Run

```bash
python main_train.py
```

## Outputs

The training pipeline writes artifacts to `OUTPUT_DIR`. The model saves both the **best** checkpoint (based on your configured evaluation metric) and the **last** checkpoint at the end of training.

