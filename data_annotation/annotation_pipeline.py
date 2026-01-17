import os
from PIL import Image

INPUT_DIR = "data"
OUTPUT_DIR = "prepared_data"
TARGET_SIZE = 720 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_image(path_in, path_out):
    img = Image.open(path_in).convert("RGB")
    w, h = img.size
    scale = TARGET_SIZE / max(w, h)
    new_size = (int(w * scale), int(h * scale))
    img = img.resize(new_size, Image.LANCZOS)
    img.save(path_out)

file_list = []

print("INPUT_DIR =", os.path.abspath(INPUT_DIR))
print("Содержимое:", os.listdir(INPUT_DIR))

for ftype in os.listdir(INPUT_DIR):
    ftype_path = os.path.join(INPUT_DIR, ftype)
    if not os.path.isdir(ftype_path):
        continue

    for pose in os.listdir(ftype_path):
        pose_path = os.path.join(ftype_path, pose)
        if not os.path.isdir(pose_path):
            continue

        output_pose_dir = os.path.join(OUTPUT_DIR, ftype, pose) 
        os.makedirs(output_pose_dir, exist_ok=True)

        for fname in os.listdir(pose_path):

            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                src = os.path.join(pose_path, fname)
                dst = os.path.join(output_pose_dir, fname)
                normalize_image(src, dst)
                file_list.append(dst)

print("Готово. Количество изображений:", len(file_list))

