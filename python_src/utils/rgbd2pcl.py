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

sys.path.append('..')
from utils.dataloader import DatasetRGBD

import open3d as o3d

def rgbd2pcl(datadir):
    RGBD_data = DatasetRGBD(datadir)

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    for timestamp, rgb_frame, depth_frame in tqdm(RGBD_data):
        rgb_o3d = o3d.geometry.Image(rgb_frame)
        depth_o3d = o3d.geometry.Image(depth_frame)
        rgbd_o3d = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=5000, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_o3d, rgb_camera_intrinsic)
        o3d.io.write_point_cloud(datadir + 'pcl/{}.xyzrgb'.format(timestamp), pcd, print_progress=True)

if __name__=='__main__':
    rgbd2pcl('../../datasets/rgbd_dataset_freiburg1_desk/')