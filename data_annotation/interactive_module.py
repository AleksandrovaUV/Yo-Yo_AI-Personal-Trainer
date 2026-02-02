import os 
import json 
import cv2 
import numpy as np

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


