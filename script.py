#!/usr/bin/python3
import os
import cv2

import mrcnn.config
import mrcnn
from mrcnn.visualize import random_colors
from mrcnn.model import MaskRCNN

class MaskRCNNConfig(mrcnn.config.Config):
    NAME = "coco_pretrained_model_config"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.8
    NUM_CLASSES = 81

import mrcnn.utils
DATASET_FILE = "mask_rcnn_coco.h5"
if not os.path.exists(DATASET_FILE):
    mrcnn.utils.download_trained_weights(DATASET_FILE)

model = MaskRCNN(mode="inference", model_dir="logs", config=MaskRCNNConfig())
model.load_weights(DATASET_FILE, by_name=True)

import numpy as np
def visualize_detections(image, masks, boxes, class_ids, scores):
    bgr_image = image.copy()
    font = cv2.FONT_HERSHEY_DUPLEX
    size = 1
    width = 1
    color = (255, 255, 255)
    for i in range(boxes.shape[0]):        
        y1, x1, y2, x2 = boxes[i]       
        if int(class_ids[i])!=1:
            continue
            
        text = "Human: {:.3f}".format(scores[i])
        cv2.rectangle(bgr_image, (x1, y1), (x2, y2), color, width)
        cv2.putText(bgr_image, text, (x1, y1-20), font, size, color, width)
    return bgr_image

IMAGE_DIR = os.path.join(os.getcwd(), "images")
files = os.listdir(IMAGE_DIR)
for filename in files:
    if 'done_' in filename:
        files.remove(filename[5:]) if files.count(filename[5:]) else None
        continue
    image = cv2.imread(os.path.join(IMAGE_DIR, filename))
    rgb_image = image[:, :, ::-1]
    detections = model.detect([rgb_image])[0]
    output_image = visualize_detections(image,
                                        detections['masks'],
                                        detections['rois'],
                                        detections['class_ids'],
                                        detections['scores'])
    cv2.imwrite(os.path.join(IMAGE_DIR, 'done_'+filename),output_image)