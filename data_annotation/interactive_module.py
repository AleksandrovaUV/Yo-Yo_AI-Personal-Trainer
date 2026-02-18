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

KEYPOINT_NAMES = [ # 33 POINTS  
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


def save_annotations(res_path, points, width, height, img_path, keypoint_name):

    if os.path.exists(res_path): 
        with open(res_path, "r", encoding="utf-8") as f: 
            data = json.load(f)
    
    else: data = []

    normed = []

    if len(points) < 3:
        print('None points added.')
        return
    
    else:
        print(points)
        for x, y, z in points: 
            if x is None or y is None: normed.append(None) 
            else: normed.append([x/width, y/height, z])

        data.append({"image": img_path,
        "keypoint name": keypoint_name,
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

    '''
    draw_buttons: drawing 'save', 'next', 'quit' buttons with basic shapes
    click_button: checking if the mouse position is on the button and if yes -- on which one
    mouse_callback:
    run:
    '''

    def __init__(self, img_path, flag = True):

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
        if preannot is not None and flag is not False:
            self.points = preannot

        self.zoom = 1.0
        self.min_zoom = 0.5
        self.max_zoom = 3.0

        self.offsetx = 0 # a parametr that projects mouth clicks onto the xoomed canvas
        self.offsety = 0


    
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

    def mouse_callback(self, event, x, y, flags, state):

        if event == cv2.EVENT_LBUTTONDOWN and y > self.h - BUTTON_HEIGHT: # => a button was pressed
            button = self.click_button(x,y)
            if button is not None:
                state["clicked"] = button
            return

        if event == cv2.EVENT_MOUSEHWHEEL:
            if flags > 0: # zooming in
                self.zoom = min(self.zoom + 0.1, self.max_zoom)
            else: self.zoom = max(self.min_zoom, self.zoom - 0.1)
            return
            
        if event == cv2.EVENT_LBUTTONDOWN: # => an image was pressed (keypoint is to be initiated)

            dists = []
            for px, py, pz in self.points:
                if px is None:
                    dists.append(1e9)
                else:
                    dists.append(np.linalg.norm(np.array([x, y]) - np.array([px, py])))

            idx = int(np.argmin(dists))
            if dists[idx] < 20:
                self.act = idx
                self.dragging = True
                
            else:
                for n, (px, py, pz) in enumerate(self.points):
                    if px is None:
                        self.points[n] = [x, y, 0]
                        self.act = n
                        self.dragging = True
                        break

        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.act is not None:
            self.points[self.act][0] = x
            self.points[self.act][1] = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def run(self):

        window = 'Yo-Yo Annotator'
        cv2.namedWindow(window)
        
        state = {"clicked": None}
        cv2.setMouseCallback(window, self.mouse_callback, state)

        while True:

            canvas = self.img.copy()

            canvas = self.draw_buttons(canvas)

            
            if self.act is not None:
                cv2.putText(canvas, f"Active: {KEYPOINT_NAMES[self.act]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)



            for i, (x, y, z) in enumerate(self.points):
                if x is not None:
                    color = POINT_COLOR_ACTIVE if i == self.act else POINT_COLOR
                    cv2.circle(canvas, (int(x), int(y)), POINT_RADIUS, color, POINT_THICKNESS)
                    cv2.putText(canvas, KEYPOINT_NAMES[i], (int(x)+10, int(y)+10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

            cv2.imshow(window, canvas)
            cv2.waitKey(20)

            if state["clicked"] == "save":
                if self.points is not None:
                    save_annotations(OUTPUT_JSON,  self.points, self.w, self.h, self.img_path, KEYPOINT_NAMES[self.act])
                    print("Saved:", self.img_path)
                else: ValueError("No keypoints found.")
                state["clicked"] = None

            elif state["clicked"] == "next":
                if self.points is not None:
                    save_annotations(OUTPUT_JSON,  self.points, self.w, self.h, self.img_path, KEYPOINT_NAMES[self.act])
                    print("Saved:", self.img_path)
                else: ValueError("No keypoints found.")
                cv2.destroyWindow(window)
                return "next"
            
            
            elif state["clicked"] == "quit":
                cv2.destroyWindow(window)
                return "quit"


def main():

    print('Free annotation <f> or error-based <e>?')
    ans = input().lower()

    print('Load pre-made annotations (if exist)? <y/n>')
    inp = input().lower()
    if inp == 'y': flag = True
    else: flag = False


    if ans == 'f':
        INPUT_FILE = choose_image_file(INPUT_DIR)
        print(f"Chosen file: {INPUT_FILE}")

        if not INPUT_FILE: 
            print("No file chosen.") 
            return
        
        all_dir_files = []

        for root, dirs, files in os.walk(INPUT_DIR):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    all_dir_files.append(os.path.join(root, f))

        all_dir_files = sorted(all_dir_files)

        start_path = os.path.relpath(INPUT_FILE)
        all_paths = [os.path.relpath(f) for f in all_dir_files]
        index = all_paths.index(start_path)

        while index < len(all_paths):

            poser = Poser(all_paths[index], flag)
            button_click = poser.run()

            if button_click == "next":
                index += 1
                continue

            if button_click == "quit":
                print("Exiting the annotator.")
                break

            else:
                break
        
        print("App closed")

    elif ans == 'e':
        '''
        Goes through a list of images with the 'anomaly' flag
        '''
        print('this part of the program is still a wip')


if __name__ == "__main__":
    main()
