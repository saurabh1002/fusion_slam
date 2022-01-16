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
from utils.dataloader import DatasetGroundTruth
from utils.dl.filter_object_masks import filter_object_masks


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

    data = DatasetGroundTruth(datadir)

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    detection_threshold = 0.8
    edge_threshold = 20
    min_mask_area = 50 * 50

    object_tsdf_dict = {}

    for n, [_, rgb_cv2, depth_cv2, origin, rot] in tqdm(enumerate(data)):
        rgb_frame_o3d = o3d.geometry.Image(rgb_cv2)

        # masks(n, 480, 640), boxes(n, 2, 2), pred_cls(n,)
        pred_cls, boxes, masks = filter_object_masks(
            data.rgb_paths[n], detection_threshold, edge_threshold, min_mask_area
        )
        T = compute_SE3_tf(origin, rot)

        for pred, mask in zip(pred_cls, masks):
            if pred not in object_tsdf_dict.keys():
                object_tsdf_dict[pred] = o3d.pipelines.integration.ScalableTSDFVolume(
                                         voxel_length=0.01,
                                         sdf_trunc=0.05,
                                         color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
                                        )
    
            depth_frame_o3d = o3d.geometry.Image(depth_cv2 * mask)
            rgbd_frame = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_frame_o3d,
                depth_frame_o3d,
                depth_scale=5000,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False,
            )

            object_tsdf_dict[pred].integrate(rgbd_frame, rgb_camera_intrinsic, T)

    for key in tqdm(object_tsdf_dict.keys()):
        mesh = object_tsdf_dict[key].extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([mesh], key + "tsdf volume")
        o3d.io.write_triangle_mesh(
            "../../results/" + key + "_tsdf_known_poses.ply",
            mesh)