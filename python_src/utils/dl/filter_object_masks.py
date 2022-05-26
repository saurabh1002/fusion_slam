#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from typing import Tuple
import numpy as np

from tqdm import tqdm

import sys
sys.path.append("..")
from dataloader import DatasetRGBD

import torch
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print(f"torch: {TORCH_VERSION}; cuda: {CUDA_VERSION}")

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os, cv2

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.projects.panoptic_deeplab import (
    PanopticDeeplabDatasetMapper,
    add_panoptic_deeplab_config,
)

def near_edge(mask: np.ndarray, edge_threshold) -> bool:
    """Check if the prediction mask is near the image boundary

    Arguments
    ---------
    - mask: A boolean array indicating the object mask within the image
    - edge_threshold: number of pixels to define "nearness" to the image boundary
    """
    return (
        (mask[:, :edge_threshold].any() == True)
        or (mask[:edge_threshold, :].any() == True)
        or (mask[:, -edge_threshold:].any() == True)
        or (mask[-edge_threshold:, :].any() == True)
    )


def filter_object_masks(
    rgb_frame_path: str,
    depth_frame_path: str,
    predictor,
    class_names: str,
    edge_threshold: int,
    min_mask_area: int,
) -> Tuple[list, list, list]:
    """Filter the object masks detected by the network based on their size and closeness to the image boundary

    Arguments
    ---------
    - rgb_frame_path: Path to the RGB image file
    - detection_threshold [0, 1]: Threshold for the network assigned detection scores
    - edge_threshold: number of pixels to define "nearness" to the image boundary
    - min_mask_area: minimum area of the mask (number of pixels) to retain the detections

    Returns
    -------
    - filtered_predictions: A list containing the prediction classes for the filtered masks
    - filtered_boxes: Bounding boxes of the filtered object masks
    - filtered_masks: Object masks that pass the filtering
    """
    im = cv2.imread(rgb_frame_path)
    outputs = predictor(im)
    predictions = outputs["instances"].to("cpu")
    boxes = predictions.pred_boxes.tensor.cpu().numpy() if predictions.has("pred_boxes") else None
    scores = predictions.scores.numpy() if predictions.has("scores") else None
    
    classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
    labels = np.array([class_names[i] for i in classes])

    if predictions.has("pred_masks"):
        masks = np.asarray(predictions.pred_masks)
    else:
        masks = None
    
    masks_path = os.path.splitext(rgb_frame_path)[0]
    depth_image = cv2.imread(depth_frame_path, cv2.CV_16UC1)
    try:
        os.mkdir(masks_path)
        os.mkdir(masks_path + '/masks')
    except FileExistsError:
        pass
    np.savetxt(masks_path + "/class_labels.txt", labels, fmt="%s")
    np.savetxt(masks_path + "/bboxes.txt", boxes)
    np.savetxt(masks_path + "/scores.txt", scores)
    for i, mask in enumerate(masks):
        cv2.imwrite(masks_path + '/masks/' + str(i) + '.png', mask * depth_image)

    filtered_predictions = []
    filtered_masks = []
    filtered_boxes = []

    for pred, mask, box in zip(labels, masks, boxes):
        if (not near_edge(mask, edge_threshold)) and (np.sum(mask) >= min_mask_area):
            filtered_masks.append(mask)
            filtered_boxes.append(box)
            filtered_predictions.append(pred)

    return filtered_predictions, filtered_boxes, filtered_masks

def main():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    class_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

    predictor = DefaultPredictor(cfg)
    IMAGE_DIR = "../../../datasets/rgbd_dataset_freiburg1_desk/"

    rgbd_data = DatasetRGBD(IMAGE_DIR)

    edge_threshold = 20
    min_mask_area = 50 * 50
    for n in range(len(rgbd_data)):
        # masks(n, 480, 640), boxes(n, 4), pred_cls(n,)
        pred_cls, boxes, masks = filter_object_masks(
            rgbd_data.rgb_paths[n], rgbd_data.depth_paths[n], predictor, class_names, edge_threshold, min_mask_area
        )

if __name__=='__main__':
    main()
