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
import matplotlib.pyplot as plt

import open3d as o3d

print(o3d.__version__)
if __name__=='__main__':
    IMAGE_DIR = "../../datasets/rgbd_dataset_freiburg1_desk/"

    ASSOCIATIONS_PATH = "../../datasets/rgbd_dataset_freiburg1_desk/associations_rgbd.txt"

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    with open(ASSOCIATIONS_PATH, 'r') as f:
        for line in f.readlines():
            _, rgb_path, _, depth_path = line.rstrip("\n").split(' ')

            rgb_image = o3d.io.read_image(IMAGE_DIR + rgb_path)
            depth_image = o3d.io.read_image(IMAGE_DIR + depth_path)
            
            target_rgbd_image = o3d.geometry.RGBDImage.create_from_tum_format(rgb_image, depth_image)
            
            pcl = o3d.geometry.PointCloud.create_from_rgbd_image(target_rgbd_image, rgb_camera_intrinsic)
            pcl.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

            # pcl.estimate_normals()

            # Save pointcloud coordinates
            flag = o3d.io.write_point_cloud(IMAGE_DIR + 'pointclouds/' + rgb_path[4:-4] + '.xyz', pcl, True, True, True)
            o3d.visualization.draw_geometries([pcl], 'TUM PCL', point_show_normal=True)

            # Alpha Shape from PCL
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcl, 0.03)
            # mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

            # Voxelization from PCL
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcl, voxel_size=0.01)
            o3d.visualization.draw_geometries([voxel_grid])