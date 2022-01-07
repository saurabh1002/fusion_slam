#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import os
import cv2
import numpy as np
from tqdm import tqdm

import open3d as o3d

if __name__ == "__main__":
    directory = "../../results/"

    for filename in os.scandir(directory):
        if filename.is_file():
            print(filename.path)
            pcl = o3d.io.read_point_cloud(filename.path)
            o3d.visualization.draw_geometries([pcl], str(filename))
