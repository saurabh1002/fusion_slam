#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import os
from typing import Tuple
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from PIL import Image
import torch
import torchvision
from torchvision import transforms as T

import sys

sys.path.append("..")
from utils.dataloader import DatasetRGBD
from utils.dl.mask_rcnn import get_prediction
from utils.plots.opencv import random_colour_masks, show_object_masks


model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def get_prediction(
    img_path: str, threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    pred = model([img])
    pred_score = list(pred[0]["scores"].detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (pred[0]["masks"] > 0.5).squeeze().detach().cpu().numpy()
    pred_class = [
        COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]["labels"].numpy())
    ]
    pred_boxes = [
        [(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]
        for i in list(pred[0]["boxes"].detach().numpy())
    ]
    masks = masks[: pred_t + 1]
    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    return np.array(masks), np.array(pred_boxes), np.array(pred_class)


def instance_segmentation_api(
    img_path: str, save_path: str, threshold: float = 0.5
) -> None:
    masks, boxes, pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    show_object_masks(img, pred_cls, boxes, masks)


if __name__ == "__main__":
    IMAGE_DIR = "../../datasets/rgbd_dataset_freiburg1_desk/"
    try:
        os.makedirs(IMAGE_DIR + "rgb_masks/")
    except FileExistsError:
        pass
    for image_name in os.listdir(IMAGE_DIR + "rgb/"):
        if image_name.endswith(".png") or image_name.endswith(".jpg"):
            instance_segmentation_api(
                IMAGE_DIR + "rgb/" + image_name,
                IMAGE_DIR + "rgb_masks/" + image_name,
                threshold=0.85,
            )
        else:
            print("Image extension not supported")
