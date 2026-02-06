import os 
import json 
import cv2 
import numpy as np
from tkinter import Tk, filedialog


'''
settings for the project
'''

INPUT_DIR = "prepared_data" 

OUTPUT_JSON = "data_annotation/manual_annotation.json"

POINT_RADIUS = 6 
POINT_COLOR = (255, 197, 128) 
POINT_COLOR_ACTIVE = (0, 255, 255) 
POINT_THICKNESS = -1

PREANNOTATIONS_JSON = 'data_annotation\preannotations.json'

'''
UTILITIES:

load_annotations: loads existing annotations (if they exist) from a file


'''

def load_annotations(path):
    if path is None or not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def choose_image_file(folder):

    root = Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        title="Выберите изображение",
        initialdir=folder,
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp")]
    )

    file_path = os.path.relpath(file_path)

    return file_path



'''
Poser class allows for:

> 
> 

'''

class Poser():

    def __init__(self, img_path ):

        self.img_path = img_path
        self.img = cv2.imread(img_path)
        if self.img is None: raise ValueError(f'Не удалось загрузить изображение {img_path}')

        self.dragging = False
        self.points = [[None, None, 0] for _ in range(33)]
        self.act = None

    def mouse_callback(self, x, y, event, flag, param):

        if event == cv2.EVENT_LBUTTONDBLCLK:

            for n, (px, py, pz) in enumerate(self.points):
                if px is None:
                    self.points[n] = [x, y, 0]
                    self.act = n
                    self.dragging = True
                    break
            
    def run(self):

        window = 'Yo-Yo Annotator'
        cv2.namedWindow(window)
        cv2.setMouseCallback(window, self.mouse_callback)

        while True:
            cv2.imshow(window)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('s'): 
                cv2.destroyWindow(window)
                return self.points

            if key == ord('q'): 
                cv2.destroyWindow(window)
                return

def main():

    # if points is None:
    #     print('Разметка отменена')
    #     return

    INPUT_FILE = choose_image_file(INPUT_DIR)
    print(INPUT_FILE)

    if not INPUT_FILE: 
        print("Файл не выбран") 
        return
    
    # annotations = [{"image": INPUT_FILE,
    # "keypoints": norm_points}]
    
    # with open(r"data_annotation/annotations.json", "w") as f: 
    #     json.dump(annotations, f, indent=2) 

if __name__ == "__main__":
    main()
