import cv2 as cv
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# путь к модели .task
model_path = r"model_0.0\pose_landmarker_full.task"

# создаём детектор позы
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=False
)
detector = vision.PoseLandmarker.create_from_options(options)

# функция рисования
def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = rgb_image.copy()

    if detection_result.pose_landmarks:
        for landmarks in detection_result.pose_landmarks:
            for lm in landmarks:
                h, w, _ = annotated_image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv.circle(annotated_image, (cx, cy), 20, (0, 255, 0), -1)

    return annotated_image

# загрузка изображения
img = cv.imread(r"model_0.0\pose.png")
rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# создаём входной объект
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

# детекция
result = detector.detect(mp_image)

# рисуем
annotated = draw_landmarks_on_image(img, result)

cv.imwrite(r"model_0.0\mediapipe_annotated_pose.png", annotated)
