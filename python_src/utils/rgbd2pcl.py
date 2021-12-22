#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import sys
from tqdm import tqdm

sys.path.append("..")
from utils.dataloader import DatasetRGBD

import open3d as o3d
import numpy as np


def truncate_pcl_3d(
    pointcloud: o3d.geometry.PointCloud, central_percentile: float
) -> (o3d.geometry.PointCloud):

    lower = 50 - central_percentile / 2
    upper = 50 + central_percentile / 2
    for dim in range(3):
        points = np.array(pointcloud.points)
        idx = np.where(
            (np.percentile(points[:, dim], lower) < points[:, dim])
            & (points[:, dim] < np.percentile(points[:, dim], upper))
        )
        pointcloud = pointcloud.select_by_index(list(idx[0]))

    return pointcloud


def rgbd2pcl(datadir):
    RGBD_data = DatasetRGBD(datadir)

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    for timestamp, rgb_frame, depth_frame in tqdm(RGBD_data):
        rgb_o3d = o3d.geometry.Image(rgb_frame)
        depth_o3d = o3d.geometry.Image(depth_frame)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=5000,
            depth_trunc=3,
            convert_rgb_to_intensity=False,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_o3d, rgb_camera_intrinsic
        )
        pcd = truncate_pcl_3d(pcd, 90)
        # o3d.visualization.draw_geometries([pcd], "TUM desk voxel map")
        o3d.io.write_point_cloud(
            datadir + "pcl/{}.xyzrgb".format(timestamp), pcd, print_progress=True
        )


if __name__ == "__main__":
    rgbd2pcl("../../datasets/rgbd_dataset_freiburg1_desk/")
