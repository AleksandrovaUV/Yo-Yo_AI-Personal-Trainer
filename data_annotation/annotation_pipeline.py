import cv2 as cv
import numpy as np
import mediapipe as mp
import os
import json

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from iterative_module import check_pose



model_path = r"model_0.0\pose_landmarker_full.task"

INPUT_DIR = r"prepared_data" 
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


base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

annotations = []

def list_images(path): # detecting an image folder or a directory 
    items = os.listdir(path)
    if any(os.path.isfile(os.path.join(path, x)) for x in items):
        return [os.path.join(path, x) for x in items if x.lower().endswith((".jpg",".jpeg",".png"))]

    result = []
    for x in items:
        sub = os.path.join(path, x)
        if os.path.isdir(sub):
            result.extend(list_images(sub))
    return result

all_images = list_images(INPUT_DIR)


for src in all_images:

    img = cv.imread(src)
    rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    keypoints = []

    if not result.pose_landmarks:
        print("No pose in: ", src)
        continue

    for landmarks in result.pose_landmarks[0]:
        keypoints.append([landmarks.x, landmarks.y])

    correction = check_pose(keypoints)
    
    annotations.append({
        "image": src, 
        "keypoints": keypoints,
        "status": correction["status"],
        "issues": correction["issues"]
    })

    if correction["status"] != "OK":
        print(f"[{correction['status']}] {src}")

with open(r"data_annotation/preannotations.json", "w") as f: 
    json.dump(annotations, f, indent=2) 

try:
    detector.close()
except:
    pass


print("Done.")