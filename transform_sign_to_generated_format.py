import numpy as np
from sys import argv
import json
import os
from PIL import Image

DATASET_BASE_PATH = argv[1]
ORIGINAL_WIDTH_HEIGHT = 600
TARGET_WIDTH_HEIGHT = 256
IMAGE_SCALE_FACTOR = TARGET_WIDTH_HEIGHT / ORIGINAL_WIDTH_HEIGHT
TARGET_IMG_MODE = 'RGB'
SPLIT_KIND = 'train'  # 'validation'
TARGET_DIR = SPLIT_KIND + '_resized'
if not os.path.isdir(os.path.join(DATASET_BASE_PATH, TARGET_DIR)):
    os.mkdir(os.path.join(DATASET_BASE_PATH, TARGET_DIR))
    # recreate substructure of 'train' directory
    for dir, _, _ in os.walk(os.path.join(DATASET_BASE_PATH, SPLIT_KIND)):
        if not os.path.isdir(dir.replace(SPLIT_KIND, TARGET_DIR)):
            os.mkdir(dir.replace(SPLIT_KIND, TARGET_DIR))

train_gt = json.load(open(os.path.join(DATASET_BASE_PATH, f'{SPLIT_KIND}.json')))

# train_gt[0] := {'file_name': 'train/0000/0000/0.png', 'text': ['Prewitt'], 'bounding_boxes': [[19, 253, 193, 313]]}

def format_bounding_boxes(input_bbs):
    output_bbs = []

    for bb in input_bbs:
        left, top, right, bottom = bb
        # coordinate system origin is in top-left corner
        output_bbs.append([
            [int(left * IMAGE_SCALE_FACTOR), int(top * IMAGE_SCALE_FACTOR)],
            [int(right * IMAGE_SCALE_FACTOR), int(bottom * IMAGE_SCALE_FACTOR)]
        ])

    return output_bbs

bounding_boxes = []

with open(os.path.join(DATASET_BASE_PATH, TARGET_DIR, 'image_locations.txt'), 'w') as image_locations_txt:

    for sample_meta_information in train_gt:
        bounding_boxes.append(format_bounding_boxes(sample_meta_information['bounding_boxes']))
        path = sample_meta_information['file_name'].replace(SPLIT_KIND + '/', '')
        image_locations_txt.write(f'./{path}\n')

        img = Image.open(os.path.join(DATASET_BASE_PATH, SPLIT_KIND, path))
        if img.mode != TARGET_IMG_MODE:
            img = img.convert(TARGET_IMG_MODE)
        img = img.resize((TARGET_WIDTH_HEIGHT, TARGET_WIDTH_HEIGHT), Image.LANCZOS)
        img.save(os.path.join(DATASET_BASE_PATH, TARGET_DIR, path))

np.save(os.path.join(DATASET_BASE_PATH, TARGET_DIR, 'bounding_boxes.npy'), bounding_boxes)
