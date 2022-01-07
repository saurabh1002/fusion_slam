#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
from tqdm import tqdm

import open3d as o3d

import sys

sys.path.append("..")
from utils.dataloader import DatasetRGBD, DatasetGroundTruth
from utils.dl.mask_rcnn import get_prediction


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


if __name__ == "__main__":

    datadir = "../../datasets/rgbd_dataset_freiburg1_desk/"

    rgbd_data = DatasetRGBD(datadir)
    gt_poses = DatasetGroundTruth(datadir)

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    object_pcl_dict = {}

    for n, [timestamps, rgb_frame, depth_frame] in tqdm(enumerate(rgbd_data)):
        # if n == 1:
        #     break
        rgb_frame_o3d = o3d.geometry.Image(rgb_frame)

        # masks(n, 480, 640), boxes(n, 2, 2), pred_cls(n,)
        masks, boxes, pred_cls = get_prediction(rgbd_data.rgb_paths[n], 0.8)

        for pred, mask in zip(pred_cls, masks):
            depth_frame_o3d = o3d.geometry.Image(depth_frame * mask)
            rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_frame_o3d,
                depth_frame_o3d,
                depth_scale=5000,
                convert_rgb_to_intensity=False,
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_frame,
                rgb_camera_intrinsic,
                extrinsic=compute_SE3_tf(
                    np.array(gt_poses.origins[n]), np.array(gt_poses.rot_quat[n])
                ),
            )
            try:
                pcd = truncate_pcl_3d(pcd, 80)
            except IndexError:
                pass

            # o3d.visualization.draw_geometries([pcd], "Truncated PCD")

            # Check if objet class exixts in dictionary
            # Else create a new entry in the dictionary corresponding to the object
            try:
                pcd_tmp = object_pcl_dict[pred]
                object_pcl_dict[pred] = pcd + pcd_tmp
            except KeyError:
                object_pcl_dict[pred] = pcd

            # Occasionally downsample the pointcloud
            if (n + 1) % 10 == 0:
                object_pcl_dict[pred] = object_pcl_dict[pred].voxel_down_sample(0.0002)

    pcd = o3d.geometry.PointCloud()
    for key in object_pcl_dict.keys():
        o3d.visualization.draw_geometries([object_pcl_dict[key]], key + "voxel map")
        o3d.io.write_point_cloud(
            "../../results/" + key + "_pcl.xyzrgb",
            object_pcl_dict[key],
            print_progress=True,
        )
        pcd += object_pcl_dict[key]

    o3d.visualization.draw_geometries([pcd], "Full voxel map")
