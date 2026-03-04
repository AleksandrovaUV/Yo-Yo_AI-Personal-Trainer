import os
import json
import cv2
import numpy as np

INPUT_DIR = "prepared_data"
PREANNOT = "data_annotation/preannotations.json"
OUTPUT_DIR = "annotated_images"

os.makedirs(OUTPUT_DIR, exist_ok=True)

POINT_RADIUS = 5
POINT_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

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


def load_preannotations(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_keypoints(img, keypoints):
    h, w = img.shape[:2]

    for i, kp in enumerate(keypoints):
        if kp is None:
            continue

        x, y = kp

        px = int(x * w)
        py = int(y * h)

        cv2.circle(img, (px, py), POINT_RADIUS, POINT_COLOR, -1)
        cv2.putText(img, KEYPOINT_NAMES[i], (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1)

    return img


def main():
    data = load_preannotations(PREANNOT)

    for item in data:
        img_path = item["image"]

        full_path = os.path.join(img_path)

        if not os.path.exists(full_path):
            print("Image not found:", full_path)
            continue

        img = cv2.imread(full_path)
        if img is None:
            print("Failed to load:", full_path)
            continue

        keypoints = item["keypoints"]

        annotated = draw_keypoints(img.copy(), keypoints)

        out_name = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, out_name)

        cv2.imwrite(out_path, annotated)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
