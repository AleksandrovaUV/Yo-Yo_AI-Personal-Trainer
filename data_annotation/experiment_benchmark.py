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


IMAGE_DIR = r"prepared_data\valid\Tree"
RESULTS_JSON = "data_annotation/experiment.json"
model_path_pipe = r"model_0.0\pose_landmarker_full.task"
model_path_mov = r"movenet"
GT_JSON = r"data_annotation\manual_annotation.json"

KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]

# mapping to allign annotated points with model points

MP_MAPPING = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 17,
    "right_thumb": 18,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32
}

MOVENET_MAPPING = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16
}


def map_to_my_format(pred, mapping):
    result = np.full((len(KEYPOINT_NAMES), 2), np.nan)
    for i, name in enumerate(KEYPOINT_NAMES):
        if name in mapping:
            idx = mapping[name]
            if idx < len(pred):
                result[i] = pred[idx][:2]
    return result



    
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
            keypoints.append([landmarks.x, landmarks.y])  
        return map_to_my_format(keypoints, MP_MAPPING)
    


# movenet
class MoveNetWrapper:
    def __init__(self, model_path):
        self.model = tf.saved_model.load(model_path)
        self.infer_fn = self.model.signatures["serving_default"]

    def infer(self, img):
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        size = max(h, w)

        pad_top = (size - h) // 2
        pad_bottom = size - h - pad_top
        pad_left = (size - w) // 2
        pad_right = size - w - pad_left

        img_square = cv.copyMakeBorder(
            img_rgb, pad_top, pad_bottom, pad_left, pad_right,
            cv.BORDER_CONSTANT, value=(0, 0, 0)
        )

        img_resized = cv.resize(img_square, (256, 256))
        inp = img_resized.astype(np.int32)[None]

        outputs = self.infer_fn(input=inp)

        out = outputs["output_0"].numpy()[0, 0, :, :2]
        out = out[:, [1, 0]]  


        return map_to_my_format(out, MOVENET_MAPPING)



    

def pck(pred, real, alpha=0.05):
    mask = ~np.isnan(real[:,0]) & ~np.isnan(pred[:,0])
    if mask.sum() == 0:
        return None
    distances = np.linalg.norm(pred[mask] - real[mask], axis=1)
    return float((distances < alpha).mean())



def load_groundtruth(gt_path):
    with open(gt_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt = {}
    for item in data:
        filename = os.path.basename(item["image"])
        gt[filename] = item["keypoints"]

    return gt




def benchmark_model(name, model, image_dir, gt, predictions_only):
    times = []
    results = []

    for fname in os.listdir(image_dir):
        if not fname.lower().endswith((".jpg", ".png")):
            continue

        img_path = os.path.join(image_dir, fname)
        img = cv.imread(img_path)
        if img is None:
            continue

        start = time.time()
        pred = model.infer(img)
        elapsed = time.time() - start

        if pred is None:
            predictions_only.append({
                "model": name,
                "image": fname,
                "keypoints": None
            })

            results.append({
                "image": fname,
                "time": elapsed,
                "pck@0.05": None,
                "pck@0.10": None
            })
            continue

        pred_norm = pred.astype(float)
        predictions_only.append({
            "model": name,
            "image": fname,
            "keypoints": pred_norm.tolist()  
        })

        times.append(elapsed)

        p05 = p10 = None

        if fname in gt:
            raw = gt[fname]
            real = np.array([
                [np.nan, np.nan] if kp is None else kp
                for kp in raw
            ], dtype=float)

            L = min(len(real), len(pred_norm))
            real = real[:L]
            pred_use = pred_norm[:L]

            mask = ~np.isnan(real[:, 0]) & ~np.isnan(pred_use[:, 0])

            if mask.sum() > 0:
                distances = np.linalg.norm(pred_use[mask] - real[mask], axis=1)
                p05 = float((distances < 0.05).mean())
                p10 = float((distances < 0.10).mean())

        results.append({
            "image": fname,
            "time": elapsed,
            "pck@0.05": p05,
            "pck@0.10": p10
        })

    return {
        "model": name,
        "avg_time": float(np.mean(times)) if times else None,
        "fps": float(1 / np.mean(times)) if times else None,
        "per_image": results
    }



def main():
    gt = load_groundtruth(GT_JSON)

    models = [
        ("MediaPipe", MediaPipeWrapper(model_path_pipe)),
        ("MoveNet", MoveNetWrapper(model_path_mov))
    ]

    all_results = []
    predictions_only = []

    for name, model in models:
        all_results.append(
            benchmark_model(name, model, IMAGE_DIR, gt, predictions_only)
        )

    with open("data_annotation/experiment.json","w",encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open("data_annotation/experiment_models.json","w",encoding="utf-8") as f:
        json.dump(predictions_only, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

