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
from matplotlib import pyplot as plt

import open3d as o3d

from mask_rcnn import get_prediction

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

    return np.linalg.inv(T)


if __name__=='__main__':

    DATASET_PATH = '../../datasets/rgbd_dataset_freiburg1_desk/'
    
    with open(DATASET_PATH + 'associations_rgbd.txt', 'r') as f:
        rgb_depth_mapping = {}
        for line in tqdm(f.readlines(), "Reading RGB and depth frame associations", colour='green'):
            _, rgb_path, _, depth_path = line.rstrip("\n").split(' ')
            rgb_depth_mapping[rgb_path] = depth_path

    with open(DATASET_PATH + 'associations_gt.txt', 'r') as f:
        rgb_gt_mapping = {}
        for line in tqdm(f.readlines(), "Reading Ground truth poses", colour='green'):
            _, rgb_path, _, tx, ty, tz, qx, qy, qz, qw = line.rstrip("\n").split(' ')
            rgb_gt_mapping[rgb_path] = np.array([float(tx), float(ty), float(tz), float(qw), float(qx), float(qy), float(qz)])

    rgb_frames_path = list(rgb_depth_mapping.keys())
    depth_frames_path = list(rgb_depth_mapping.values())
    groundtruth_poses = np.array([rgb_gt_mapping[rgb_frames_path[i]] for i in range(len(rgb_frames_path))])

    num_frames = len(rgb_frames_path)
    gt_SE3_tf = np.array([compute_SE3_tf(groundtruth_poses[n, :3], groundtruth_poses[n, 3:]) for n in tqdm(range(num_frames), 
        "Generating SE3 tf for ground truth poses", colour='green')])

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)
    
    object_pcl_dict = {}
    # masks(n, 480, 640), boxes(n, 2, 2), pred_cls(n,)
    for n in tqdm(range(num_frames),"Computing and stitching pointcloud from rgbd frames", colour='green'):
        rgb_frame = o3d.io.read_image(DATASET_PATH + rgb_frames_path[n])
        depth_frame_cv = cv2.imread(DATASET_PATH + depth_frames_path[n], cv2.CV_16UC1)

        masks, boxes, pred_cls = get_prediction(DATASET_PATH + rgb_frames_path[n], 0.8)
        num_objects = masks.shape[0]

        for i in range(num_objects):
            depth_frame = o3d.geometry.Image(depth_frame_cv * masks[i])
            rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_frame, depth_frame, depth_scale=5000, convert_rgb_to_intensity=False)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_frame, rgb_camera_intrinsic, extrinsic=gt_SE3_tf[n])

            try:
                pcd_tmp = object_pcl_dict[pred_cls[i]]
                object_pcl_dict[pred_cls[i]] = pcd + pcd_tmp
            except KeyError:
                object_pcl_dict[pred_cls[i]] = pcd
            if ((n+1) % 10 == 0):
                object_pcl_dict[pred_cls[i]] = object_pcl_dict[pred_cls[i]].voxel_down_sample(0.0002)

    pcd = o3d.geometry.PointCloud()
    for key in object_pcl_dict.keys():
        print(pcd, key, object_pcl_dict[key])
        o3d.visualization.draw_geometries([object_pcl_dict[key]], 'TV voxel map')
        o3d.io.write_point_cloud(DATASET_PATH + 'results/' + key + '_pcl.xyzrgb', object_pcl_dict[key], print_progress=True)
        pcd += object_pcl_dict[key]

    o3d.visualization.draw_geometries([pcd], 'TV voxel map')