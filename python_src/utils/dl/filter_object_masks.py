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
from utils.dataloader import DatasetRGBD
from utils.dl.mask_rcnn import get_prediction
from utils.plots.opencv import show_object_masks


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
    detection_threshold: float,
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
    masks, boxes, pred_cls = get_prediction(rgb_frame_path, detection_threshold)

    filtered_predictions = []
    filtered_masks = []
    filtered_boxes = []

    for pred, mask, box in zip(pred_cls, masks, boxes):
        if (not near_edge(mask, edge_threshold)) and (np.sum(mask) >= min_mask_area):
            filtered_masks.append(mask)
            filtered_boxes.append(box)
            filtered_predictions.append(pred)

    return filtered_predictions, filtered_boxes, filtered_masks


if __name__ == "__main__":

    datadir = "../../datasets/rgbd_dataset_freiburg1_desk/"

    rgbd_data = DatasetRGBD(datadir)

    detection_threshold = 0.8
    edge_threshold = 20
    min_mask_area = 50 * 50
    for n, [_, rgb_frame, _] in tqdm(enumerate(rgbd_data)):
        # masks(n, 480, 640), boxes(n, 2, 2), pred_cls(n,)
        pred_cls, boxes, masks = filter_object_masks(
            rgbd_data.rgb_paths[n], detection_threshold, edge_threshold, min_mask_area
        )
        show_object_masks(rgb_frame, pred_cls, boxes, masks)
