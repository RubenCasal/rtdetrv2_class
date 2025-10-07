**Related documentation**

* [Inference](./inference.md)
* [Evaluation metrics](./evaluation_metrics.md)
* [Data augmentation](./data_augmentation.md)
* [Dataset transformation](./dataset_transformation.md)

# Inference (image) — RT‑DETRv2

This script runs single‑image inference with your trained RT‑DETRv2 checkpoint and draws detections.

## What it does

* Loads a checkpoint from `CHECKPOINT` in **inference** mode.
* Reads an image from `IMAGE_PATH`.
* Runs `predict(image, class_thresholds=...)` and prints wall‑clock latency (`TIME` in ms) and raw results.
* Renders boxes/labels with `draw_results(...)` and displays the annotated image.

## How to run

```bash
# activate the repo environment
conda activate <your_env_name>

# run the script
python main_infer_image.py
```

## Configure

* `CHECKPOINT`: path to your trained checkpoint directory (e.g., `./rtdetr-v2-r50-cppe5-finetune-2/checkpoint-500`).
* `IMAGE_PATH`: path to the input image (e.g., `image.png`).
* `CLASS_THRESHOLDS`: per‑class confidence thresholds (required). Example:

  ```python
  CLASS_THRESHOLDS = {0: 0.4, 1: 0.4, 2: 0.4}  # person, vehicle, boat
  ```

## Output

* Console prints total inference time and the filtered detections dictionary:

  * `scores`: confidences (tensor)
  * `labels`: class ids (tensor)
  * `boxes`: `[x_min, y_min, x_max, y_max]` in pixels (tensor)
* A window pops up with the annotated image. To save it instead of (or in addition to) showing it:

  ```python
  annotated_image.save("pred.png")
  ```
