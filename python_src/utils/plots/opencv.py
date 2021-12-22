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
import random
import cv2

import matplotlib.pyplot as plt


def random_colour_masks(image: np.ndarray) -> np.ndarray:
    colors = [
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [0, 255, 255],
        [255, 255, 0],
        [255, 0, 255],
        [80, 70, 180],
        [250, 80, 190],
        [245, 145, 50],
        [70, 150, 250],
        [50, 190, 190],
    ]
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    r[image == 1], g[image == 1], b[image == 1] = colors[random.randrange(0, 10)]
    colored_mask = np.stack([r, g, b], axis=2)
    return colored_mask


def show_object_masks(
    img, pred_cls, boxes, masks, save_flag: bool = False, save_path: str = None
) -> None:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        rgb_mask = random_colour_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(255, 0, 0), thickness=1)
        cv2.putText(
            img,
            pred_cls[i],
            boxes[i][1],
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            thickness=1,
        )
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.waitforbuttonpress()
    plt.xticks([])
    plt.yticks([])
    plt.close()

    if save_flag:
        plt.savefig(save_path)
