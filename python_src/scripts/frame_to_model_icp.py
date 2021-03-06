#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import numpy as np
import scipy.spatial.transform as tf

from tqdm import tqdm

import sys

sys.path.append("..")
from utils.dataloader import DatasetRGBD

import open3d as o3d

def odom_from_SE3(t: float, TF: np.ndarray) -> (list):
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:3, :3]).as_quat()
    return list(np.r_[t, origin, rot_quat])


if __name__ == "__main__":

    datadir = "../../datasets/rgbd_dataset_freiburg1_desk/"
    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    data = DatasetRGBD(datadir)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    I = np.eye(4)
    T = np.eye(4)
    poses = []

    t, rgb_cv2, depth_cv2 = data[0]
    rgb_o3d = o3d.geometry.Image(rgb_cv2)
    depth_o3d = o3d.geometry.Image(depth_cv2)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_o3d,
        depth_o3d,
        depth_scale=5000,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False,
    )

    volume.integrate(
        rgbd,
        rgb_camera_intrinsic,
        I,
    )

    for i, [t, rgb_cv2, depth_cv2] in tqdm(enumerate(data)):
        rgb_o3d = o3d.geometry.Image(rgb_cv2)
        depth_o3d = o3d.geometry.Image(depth_cv2)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=5000,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )

        model_pcd = volume.extract_point_cloud()

        frame_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, rgb_camera_intrinsic, I
        )

        result = o3d.pipelines.registration.registration_icp(
            frame_pcd,
            model_pcd,
            0.05,
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        T = result.transformation

        volume.integrate(
            rgbd,
            rgb_camera_intrinsic,
            np.linalg.inv(T),
        )

        poses.append(odom_from_SE3(t, np.array(T)))

    np.savetxt("../../results/poses_model_to_frame.txt", np.array(poses))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("../../results/full_mesh_model_to_frame.ply", mesh)
    o3d.visualization.draw_geometries([mesh])
