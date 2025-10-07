import cv2
import numpy as np
import time
from PIL import Image
from rtdetrv2_class import RTDETRV2_CLASS

CHECKPOINT = "./rtdetr-v2-r50-cppe5-finetune-2/checkpoint-500"
MODE = "inference"
VIDEO_IN_PATH = "/home/rcasal/Desktop/projects/tracking_ptfue/algoritmos_tracking/input_tracker/beach_video_slowed.mp4"
VIDEO_OUT_PATH = "output_annotated2.mp4"
CLASS_THRESHOLDS = {0: 0.4, 1: 0.4, 2: 0.4}

rtdetrv2_class = RTDETRV2_CLASS(mode=MODE, checkpoint=CHECKPOINT)

cap = cv2.VideoCapture(VIDEO_IN_PATH)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el vídeo: {VIDEO_IN_PATH}")

src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
src_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
src_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out_w, out_h = (src_w, src_h)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT_PATH, fourcc, src_fps, (out_w, out_h))
if not writer.isOpened():
    cap.release()
    raise RuntimeError(f"No se pudo crear el vídeo de salida: {VIDEO_OUT_PATH}")

frame_idx = 0
t_infer_sum = 0.0
t_total_sum = 0.0

try:
    ret, warm = cap.read()
    if ret:
        pil_warm = Image.fromarray(cv2.cvtColor(warm, cv2.COLOR_BGR2RGB))
        _ = rtdetrv2_class.predict(pil_warm, class_thresholds=CLASS_THRESHOLDS)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        t0 = time.time()
        results = rtdetrv2_class.predict(pil_img, class_thresholds=CLASS_THRESHOLDS)
        t1 = time.time()

        annotated_pil = rtdetrv2_class.draw_results(pil_img, results)
        annotated_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

        writer.write(annotated_bgr)
        t2 = time.time()

        frame_idx += 1
        t_infer_sum += (t1 - t0) * 1000.0
        t_total_sum += (t2 - t0) * 1000.0

        if frame_idx % 30 == 0:
            print(f"[{frame_idx:05d}] infer: {(t1 - t0)*1000.0:.2f} ms | total: {(t2 - t0)*1000.0:.2f} ms")
finally:
    cap.release()
    writer.release()

if frame_idx > 0:
    print(f"[DONE] Frames procesados: {frame_idx}")
    print(f"[AVG ] infer: {t_infer_sum/frame_idx:.2f} ms | total: {t_total_sum/frame_idx:.2f} ms")
    print(f"[OUT ] {VIDEO_OUT_PATH}")
