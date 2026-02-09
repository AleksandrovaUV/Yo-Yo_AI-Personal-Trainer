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

POINT_RADIUS = 5
POINT_COLOR = (255, 197, 128)
POINT_COLOR_ACTIVE = (0, 255, 255)
POINT_THICKNESS = -1

BUTTON_HEIGHT = 60
BUTTON_COLOR = (60, 60, 60)
BUTTON_TEXT_COLOR = (255, 255, 255)

PREANNOTATIONS_JSON = 'data_annotation\preannotations.json'

'''
UTILITIES:

load_annotations: loads existing annotations (if they exist) from a json file
choose_image_file: opens a file manager with an ability to select a desired image easily
save_annotaions: saving new annotations to a file

'''

def load_annotations(path, img_path, width, height):
    if path is None or not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f: 
        data = json.load(f)

        filename = os.path.basename(img_path)

        for item in data:
            if item["image"].endswith(filename):
                points = []
                for (x,y,z) in item["keypoints"]:
                    px = x * width
                    py = y * height
                    points.append([px, py, z])
                return points
    
    return None


def save_annotations(res_path, points, width, height, img_path):

    if os.path.exists(res_path): 
        with open(res_path, "r", encoding="utf-8") as f: 
            data = json.load(f)
    
    else: data = []

    normed = []
    for x,y,z in points:
        normed.append([x/width, y/height, z])

    data.append({"image": img_path,
    "keypoints": normed})
    
    with open(res_path, "w") as f: 
        json.dump(data, f, indent=2) 


def choose_image_file(folder):

    root = Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename(
        title="Choose an image",
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

        '''
        :param img_path: path to an image

        img: image file
        dragging: flag to identify point dragging
        points: keypoints in an image
        buttons: switch buttons (save, next, quit)
        '''
        
        self.img_path = img_path
        self.img = cv2.imread(img_path)
        if self.img is None: raise ValueError(f'Не удалось загрузить изображение {img_path}')

        self.h, self.w = self.img.shape[:2]

        self.dragging = False
        self.points = [[None, None, 0] for _ in range(33)]
        self.act = None

        self.buttons = {}

        preannot = load_annotations(PREANNOTATIONS_JSON, img_path, self.w, self.h)
        if preannot is not None:
            self.points = preannot
    
    def draw_buttons(self, canvas):

        new_y = self.h - BUTTON_HEIGHT
        new_y2 = self.h

        # save button
        save_box = (0, new_y, self.w // 3, new_y2)
        self.buttons["save"] = save_box
        cv2.rectangle(canvas, (save_box[0], save_box[1]), (save_box[2], save_box[3]), BUTTON_COLOR, -1)
        cv2.putText(canvas, "Save", (save_box[0] + 40, new_y + 40), cv2.FONT_HERSHEY_DUPLEX, 1, BUTTON_TEXT_COLOR, 2)

        # next button
        next_box = (self.w // 3, new_y, 2* self.w // 3, new_y2)
        self.buttons["next"] = next_box
        cv2.rectangle(canvas, (next_box[0], next_box[1]), (next_box[2], next_box[3]), BUTTON_COLOR, -1)
        cv2.putText(canvas, "Next", (next_box[0] + 40, new_y + 40), cv2.FONT_HERSHEY_DUPLEX, 1, BUTTON_TEXT_COLOR, 2)


        # quit button
        quit_box = (2* self.w // 3, new_y, self.w, new_y2)
        self.buttons["quit"] = quit_box
        cv2.rectangle(canvas, (quit_box[0], quit_box[1]), (quit_box[2], quit_box[3]), BUTTON_COLOR, -1)
        cv2.putText(canvas, "Quit", (quit_box[0] + 40, new_y + 40), cv2.FONT_HERSHEY_DUPLEX, 1, BUTTON_TEXT_COLOR, 2)

        return canvas
    
    def click_button(self, x, y):
        for name, (x1, y1, x2, y2) in self.buttons.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                return name
        return None

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
        
        state = {"clicked": None}
        cv2.setMouseCallback(window, self.mouse_callback, state)

        while True:

            canvas = self.img.copy()

            canvas = self.draw_buttons(canvas)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(20) & 0xFF

            for i, (x, y, z) in enumerate(self.points):
                if x is not None:
                    color = POINT_COLOR_ACTIVE if i == self.act else POINT_COLOR
                    cv2.circle(canvas, (int(x), int(y)), POINT_RADIUS, color, POINT_THICKNESS)

            if state["clicked"] == "save":
                save_annotations(OUTPUT_JSON, self.img_path, self.points, self.w, self.h)
                print("Сохранено:", self.img_path)
                state["clicked"] = None

            elif state["clicked"] == "next":
                save_annotations(OUTPUT_JSON, self.img_path, self.points, self.w, self.h)
                print("Сохранено:", self.img_path)
                cv2.destroyWindow(window)
                return "next"
            
            
            elif state["clicked"] == "quit":
                cv2.destroyWindow(window)
                return "quit"


def main():

    print('Free <f> annotation or error-based <e>?')
    ans = input().lower()

    if ans == 'f':
        INPUT_FILE = choose_image_file(INPUT_DIR)
        print(INPUT_FILE)

        if not INPUT_FILE: 
            print("Файл не выбран") 
            return
        
        poser = Poser(INPUT_FILE)
        poser.run()

    elif ans == 'e':
        '''
        Goes through a list of images with the 'anomaly' flag
        '''
        print('this part of the program is still a wip')

    # if points is None:
    #     print('Разметка отменена')
    #     return


if __name__ == "__main__":
    main()
