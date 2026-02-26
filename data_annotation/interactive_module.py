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

MP_MAPPING = {
    "nose": 0,
    "left_eye": 2,
    "right_eye": 5,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 17,
    "right_thumb": 18,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32
}

'''
a whole list of MediaPipe keypoints (IN A FOLLOWING ORDER):

KEYPOINT_NAMES = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index"
]

'''

'''
UTILITIES:

load_annotations: loads existing annotations (if they exist) from a json file
choose_image_file: opens a file manager with an ability to select a desired image easily
save_annotaions: saving new annotations to a file
map_to_my_format: alligns chosen keypoints with preannotated ones

'''

def map_to_my_format(pred, mapping):
    result = np.full((len(KEYPOINT_NAMES), 2), np.nan)
    for i, name in enumerate(KEYPOINT_NAMES):
        if name in mapping:
            idx = mapping[name]
            if idx < len(pred):
                result[i] = pred[idx][:2]
    return result



def load_annotations(path, img_path, width, height):
    if path is None or not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.basename(img_path)

    for item in data:
        if item["image"].endswith(filename):
            points = []


            for kp in item["keypoints"]:

                if kp is None:
                    points.append([None, None])
                    continue

                if len(kp) == 3: x, y, _ = kp

                elif len(kp) == 2: x, y = kp

                else:
                    points.append([None, None])
                    continue

                points.append([x, y])
            
            mapped = map_to_my_format(points, MP_MAPPING)

            mapped[:, 0] = mapped[:,0] * width
            mapped[:, 1] = mapped[:,1] * height

            return mapped
        
    return None


def save_annotations(res_path, points, width, height, img_path):

    if os.path.exists(res_path): 
        with open(res_path, "r", encoding="utf-8") as f: 
            data = json.load(f)
    
    else: data = []

    normed = []

    if len(points) < 2:
        print('None points added.')
        return
    
    else:
        print(points)
        for x, y in points: 
            if x is None or y is None: normed.append(None) 
            else: normed.append([x/width, y/height])


    updated = False # searching for a file if it has already been created
    for item in data:
        if item["image"] == img_path:
            item["keypoints"] = normed
            updated = True
            print(f"Updated existing annotation for {img_path}")
            break

    if not updated: # if not created, create a new file
        data.append({
            "image": img_path,
            "keypoints": normed
        })
        print(f"Added new annotation for {img_path}")
    
    with open(res_path, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=2, ensure_ascii=False)


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
        self.points = [[None, None] for _ in range(23)]
        self.act = None

        self.buttons = {}

        preannot = load_annotations(PREANNOTATIONS_JSON, img_path, self.w, self.h)
        if preannot is not None and flag is not False:
            self.points = preannot

        self.zoom = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 4.0

        self.offsetx = 0 # a parametr that projects mouth clicks onto the xoomed canvas
        self.offsety = 0

        self.panning = False # image dragging
        self.panning_start = (0,0)
    
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
        

        elif event == cv2.EVENT_MOUSEHWHEEL: # => zoom initiated
            if flags > 0: # zooming in
                self.zoom = min(self.zoom + 0.1, self.max_zoom)
            else: self.zoom = max(self.min_zoom, self.zoom - 0.1)
            return
        
        elif event == cv2.EVENT_RBUTTONDOWN: # => image dragging initiated
            self.panning = True
            self.panning_start = (x,y)

        elif event == cv2.EVENT_MOUSEMOVE and self.panning is True: # => processing with dragging
            dx = x - self.panning_start[0] 
            dy = y - self.panning_start[1] 
            self.offsetx += dx 
            self.offsety += dy 
            self.panning_start = (x, y)

        elif event == cv2.EVENT_RBUTTONUP: self.panning = False # stop panning

        elif event == cv2.EVENT_RBUTTONDBLCLK:

            orig_x = int((x - self.offsetx) / self.zoom)
            orig_y = int((y - self.offsety) / self.zoom)


            for i, (px, py) in enumerate(self.points):
                if px is None:
                    continue

                if abs(orig_x - px) < 10 and abs(orig_y - py) < 10:
                    self.points[i] = [None, None]
                    if self.act == i:
                        self.act = None
                    return 

            
        elif event == cv2.EVENT_LBUTTONDOWN: # => an image was pressed (keypoint is to be initiated)

            orig_x = int((x - self.offsetx) / self.zoom) 
            orig_y = int((y - self.offsety) / self.zoom)

            dists = []
            for px, py in self.points:
                if px is None:
                    dists.append(1e9)
                else:
                    dists.append(np.linalg.norm(np.array([orig_x, orig_y]) - np.array([px, py])))

            idx = int(np.argmin(dists))
            if dists[idx] < 20:
                self.act = idx
                self.dragging = True
                
            else:
                for n, (px, py) in enumerate(self.points):
                    if px is None:
                        self.points[n] = [orig_x, orig_y]
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

            zoomed = cv2.resize(self.img, None, fx = self.zoom, fy = self.zoom)
            canvas = np.zeros_like(self.img)

            h, w =  self.h, self.w
            fh, fw = zoomed.shape[:2]

            x1 = max(0, -self.offsetx) 
            y1 = max(0, -self.offsety)
            x2 = min(fw, w - self.offsetx)
            y2 = min(fh, h - self.offsety)
            canvas_y1 = max(0, self.offsety)
            canvas_x1 = max(0, self.offsetx)
            crop_w = x2 - x1 
            crop_h = y2 - y1
            if crop_w > 0 and crop_h > 0: canvas[canvas_y1:canvas_y1 + crop_h, canvas_x1:canvas_x1 + crop_w] = zoomed[y1:y2, x1:x2]

            canvas = self.draw_buttons(canvas)

            
            if self.act is not None:
                cv2.putText(canvas, f"Active: {KEYPOINT_NAMES[self.act]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5)
                cv2.putText(canvas, f"Active: {KEYPOINT_NAMES[self.act]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                


            for i, (x, y) in enumerate(self.points):
                if x is not None:
                    fx = int(x * self.zoom + self.offsetx)
                    fy = int(y * self.zoom + self.offsety)
                    color = POINT_COLOR_ACTIVE if i == self.act else POINT_COLOR
                    cv2.circle(canvas, (int(fx), int(fy)), POINT_RADIUS, color, POINT_THICKNESS)
                    cv2.putText(canvas, KEYPOINT_NAMES[i], (int(fx)+10, int(fy)+10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)
                    

            cv2.imshow(window, canvas)
            cv2.waitKey(20)

            if state["clicked"] == "save":
                if self.points is not None:
                    save_annotations(OUTPUT_JSON,  self.points, self.w, self.h, self.img_path)
                    print("Saved:", self.img_path)
                else: ValueError("No keypoints found.")
                state["clicked"] = None

            elif state["clicked"] == "next":
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

    else: ValueError('This is not a valid answer.')

if __name__ == "__main__":
    main()
