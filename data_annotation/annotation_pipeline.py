import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# путь к модели .task
model_path = r"model_0.0\pose_landmarker_full.task"

INPUT_DIR = "prepared_data" 
KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index"
]


# детектор позы
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

annotations = []

for ftype in os.listdir(INPUT_DIR):
    ftype_path = os.path.join(INPUT_DIR, ftype)
    if not os.path.isdir(ftype_path):
        continue

    for pose in os.listdir(ftype_path):
        pose_path = os.path.join(ftype_path, pose)
        if not os.path.isdir(pose_path):
            continue

        for fname in os.listdir(pose_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")): continue

            src = os.path.join(pose_path, fname)

            img = cv.imread(src)
            rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            # создаём входной объект
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # детекция
            result = detector.detect(mp_image)

            keypoints = []

            if not result.pose_landmarks:
                print("Поза не найдена:", src)
                continue

            for landmarks in result.pose_landmarks[0]:
                keypoints.append([landmarks.x, landmarks.y, landmarks.z])
            
            annotations.append({
                "image": f"{INPUT_DIR}/{ftype}/{pose}/{fname}", 
                "keypoints": keypoints
            })

with open(r"data_annotation/preannotations.json", "w") as f: 
    json.dump(annotations, f, indent=2) 
    
print("Готово. Предразметка выполнена.")