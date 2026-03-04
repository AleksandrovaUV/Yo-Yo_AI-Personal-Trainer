'''
a full data procession pipeline, which inclused

* json to csv transformation (columns: img_name, KEYPOINT_NAMES[n], class, pose_name)
* data normalisation 
'''

import json
import os
import csv
import numpy as np

MANUAL_PATH = r"data_annotation\manual_annotation.json"
PREANNOT_PATH = r"data_annotation\preannotations.json"
OUTPUT_CSV = r"valid_data.csv"
GOAL_TYPE = "valid"


if PREANNOT_PATH is None or not os.path.exists(PREANNOT_PATH):
    raise ValueError("No data found in path", PREANNOT_PATH)

with open(PREANNOT_PATH, "r", encoding="utf-8") as f:
    data_prean = json.load(f)

if MANUAL_PATH is None or not os.path.exists(MANUAL_PATH):
    raise ValueError("No data found in path", MANUAL_PATH)

with open(MANUAL_PATH, "r", encoding="utf-8") as f:
    data_man = json.load(f)

manual_images = {item["image"] for item in data_man}

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

POSE_TO_CLASS = {"Bound_Angle": 0, 
                 "Cat": 1, 
                 "Downward-Facing_Dog": 2, 
                 "Mountain": 3, 
                 "Tree": 4, 
                 "Upward_Bow": 5, 
                 "Boat": 6, 
                 "Upward-Facing_Dog": 7 }

rows = []

count_p = 1
count_m = 1


for item in data_man:
    img_path = item["image"]
    filename = os.path.splitext(os.path.basename(img_path))[0]
    dir_type =os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    pose_name = os.path.basename(os.path.dirname(img_path))

    u_kps = item["keypoints"]

    if dir_type != GOAL_TYPE: continue

    print(f"written: {img_path} from MANUAL")
    count_m += 1

    clean_kps = []
    for kp in u_kps:
        if kp is None:
            clean_kps.append([np.nan, np.nan])
            continue

        if len(kp) == 3:
            x, y, _ = kp
        elif len(kp) == 2:
            x, y = kp
        else:
            clean_kps.append([np.nan, np.nan])
            continue

        clean_kps.append([float(x), float(y)])

    kps = np.array(clean_kps, dtype=float)

    class_id = POSE_TO_CLASS.get(pose_name, -1)

    row = {"img_name": filename,
           "pose_name": pose_name,
           "class": class_id}
    
    for i, point in enumerate(KEYPOINT_NAMES):
        row[f"{point}_x"] = kps[i,0]
        row[f"{point}_y"] = kps[i,1]


    rows.append(row)

for item in data_prean:
    img_path = item["image"]
    filename = os.path.splitext(os.path.basename(img_path))[0]
    dir_type =os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    pose_name = os.path.basename(os.path.dirname(img_path))

    u_kps = item["keypoints"]

    if img_path in manual_images: continue
    elif dir_type != GOAL_TYPE: continue

    print(f"written: {img_path} from PREANNOT")
    count_p += 1

    clean_kps = []
    for kp in u_kps:
        if kp is None:
            clean_kps.append([np.nan, np.nan])
            continue

        

        if len(kp) == 3:
            x, y, _ = kp
        elif len(kp) == 2:
            x, y = kp
        else:
            clean_kps.append([np.nan, np.nan])
            continue

        clean_kps.append([float(x), float(y)])

    kps = np.array(clean_kps, dtype=float)


    class_id = POSE_TO_CLASS.get(pose_name, -1)

    row = {"img_name": filename,
           "pose_name": pose_name,
           "class": class_id}
    
    for i, point in enumerate(KEYPOINT_NAMES):
        row[f"{point}_x"] = kps[i,0]
        row[f"{point}_y"] = kps[i,1]


    rows.append(row)


fieldnames = ["img_name", "pose_name", "class"]
for name in KEYPOINT_NAMES:
    fieldnames.append(f"{name}_x")
    fieldnames.append(f"{name}_y")
    
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames) 
    writer.writeheader() 
    writer.writerows(rows) 
    print("CSV saved:", OUTPUT_CSV)

print(count_m, count_p)
