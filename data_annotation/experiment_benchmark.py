# in this experiment we compare various HPE frameworks to highlight the one most useful for our task

import os
import time
import json
import numpy as np
import cv2 as cv
import tensorflow as tf
import tensorflow_hub as hub
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


IMAGE_DIR = "prepared_data/valid"
RESULTS_JSON = "data_annotation/experiment.json"
model_path_pipe = r"model_0.0\pose_landmarker_full.task"
model_path_mov = r"data_annotation\saved_model.pb"
GT_JSON = r"data_annotation\gt_data.json"


# mediapipe pose
class MediaPipeWrapper:
    def __init__(self, model_path):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def infer(self, img_bgr):
        rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        if not result.pose_landmarks:
            return None
        keypoints = []
        for landmarks in result.pose_landmarks[0]:
            keypoints.append([landmarks.x, landmarks.y, landmarks.z])  # z можно добавить при необходимости
        return np.array(keypoints) 

 

# movenet
class MoveNetWrapper:
    def __init__(self, model_path):
        self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")

    def infer(self, img_bgr):
        img_resized = cv.resize(img_bgr, (192, 192))
        rgb = cv.cvtColor(img_resized, cv.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(rgb, dtype = tf.int32)
        input_tensor = np.expand_dims(input_tensor, axis=0)

        output = self.model(input_tensor)
        return output["output_0"].numpy()[0, 0, :, :2]  # (17, 2)
    

def pck(predicted, real, alpha = 0.05):
    ditances = np.linalg.norm(predicted - real, axis = 1)
    is_correct = ditances < alpha
    return is_correct

def load_groundtruth(gt_path):
    if gt_path is None or not os.path.exists(gt_path): return ValueError('No such file for Ground Truth')
    with open(gt_path, "r", encoding="utf-8") as f: 
        data = json.load(f)

def benchmark_model(name, model, image_dir, gt_data):
    results = []
    times = []
    anomalies = 0
    total = 0

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img = cv.imread(os.path.join(image_dir, fname))
        if img is None:
            continue

        total += 1
        start = time.time()
        kpts = model.infer(img)
        elapsed = time.time() - start
        times.append(elapsed)

        pck05 = None
        pck10 = None

        if fname in gt_data and kpts is not None:
            gt = np.array(gt_data[fname]["keypoints"])[:, :2]
            if gt.shape == kpts.shape:
                pck05 = pck(kpts, gt, alpha=0.05)
                pck10 = pck(kpts, gt, alpha=0.10)

        results.append({
            "image": fname,
            "time": elapsed,
            "pck@0.05": pck05,
            "pck@0.10": pck10
        })

    return {
        "model": name,
        "avg_time": float(np.mean(times)),
        "fps": float(1.0 / np.mean(times)),
        "per_image": results
    }

def main():
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    gt_data = load_groundtruth(GT_JSON)

    mp_model = MediaPipeWrapper(model_path_pipe)
    mv_model = MoveNetWrapper(model_path_mov)

    all_results = []
    for name, model in [
        ("MediaPipe", mp_model),
        ("MoveNet", mv_model),
    ]:
        summary = benchmark_model(name, model, IMAGE_DIR)
        all_results.append(summary)

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
