# in this experiment we compare various HPE frameworks to highlight the one most useful for our task

import os
import time
import json
import numpy as np
import cv2 as cv

IMAGE_DIR = "prepared_data"
RESULTS_JSON = "data_annotation/experiment.json"
model_path = r"model_0.0\pose_landmarker_full.task"


import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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

