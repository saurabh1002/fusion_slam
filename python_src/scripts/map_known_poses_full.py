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

import sys

sys.path.append("..")
from utils.dataloader import DatasetPCL

import open3d as o3d


def compute_SE3_tf(translation: np.ndarray, quaternion: np.ndarray) -> (np.ndarray):
    """
    Arguments
    ---------
    - translation: [tx, ty, tz] 3d translation wrt reference frame
    - quaternion: [qw, qx, qy, qz] quaternion representing rotation wrt reference frame

    Returns
    -------
    SE3 transformation matrix combining the rotation and translation
    """
    T = np.eye(4)
    T[:3, :3] = o3d.geometry.PointCloud.get_rotation_matrix_from_quaternion(quaternion)
    T[:3, 3] = translation

    return T


if __name__ == "__main__":

    datadir = "../../datasets/rgbd_dataset_freiburg1_desk/"

    PCL_data = DatasetPCL(datadir)

    merged_pcd = o3d.geometry.PointCloud()

    for n, [timestamp, pcd, origin, rot] in tqdm(enumerate(PCL_data)):
        merged_pcd += pcd.transform(compute_SE3_tf(origin, rot))

        # Downsample poincloud occassionally to reduce redundant points and computation cost
        if n % 5 == 0:
            merged_pcd = merged_pcd.voxel_down_sample(0.005)

    voxel_map = o3d.geometry.VoxelGrid.create_from_point_cloud(
        merged_pcd, voxel_size=0.01
    )
    o3d.visualization.draw_geometries([voxel_map], "TUM desk voxel map")
    o3d.io.write_point_cloud(
        "../../results/pcl_full_map.pcd", merged_pcd, print_progress=True
    )
