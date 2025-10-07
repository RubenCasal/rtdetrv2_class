import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rtdetrv2_class import RTDETRV2_CLASS
import time

CHECKPOINT = "./rtdetr-v2-r50-cppe5-finetune-2/checkpoint-500"
MODE = "inference"
IMAGE_PATH = "image.png"

CLASS_THRESHOLDS = {
    0: 0.4,  # person
    1: 0.4,  # vehicle
    2: 0.4,  # boat
}

rtdetrv2_class = RTDETRV2_CLASS(mode="inference", checkpoint=CHECKPOINT)

image = Image.open("image.png").convert("RGB")

start_time = time.time()
results = rtdetrv2_class.predict(
    image,
    class_thresholds=CLASS_THRESHOLDS  
)
end_time = time.time()
total_time = end_time - start_time
print(f"TIME: {total_time*1000} ms")
print(results)

annotated_image = rtdetrv2_class.draw_results(image, results)

annotated_image.show()
