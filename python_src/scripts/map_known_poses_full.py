#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from tqdm import tqdm
import open3d as o3d
import numpy as np
import sys

sys.path.append("..")
from utils.dataloader import DatasetGroundTruth


if __name__ == "__main__":

    datadir = "../../datasets/rgbd_dataset_freiburg1_desk/"

    data = DatasetGroundTruth(datadir)

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.01,
        sdf_trunc=0.05,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)

    skip_frames = 1
    for _, rgb_cv2, depth_cv2, origin, rot in tqdm(data):
        rgb_o3d = o3d.geometry.Image(rgb_cv2)
        depth_o3d = o3d.geometry.Image(depth_cv2)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d,
            depth_o3d,
            depth_scale=5000,
            depth_trunc=3.0,
            convert_rgb_to_intensity=False,
        )
        
        T = np.eye(4)
        T[:3, :3] = o3d.geometry.PointCloud.get_rotation_matrix_from_quaternion(rot)
        T[:3, 3] = origin

        volume.integrate(
            rgbd,
            rgb_camera_intrinsic,
            np.linalg.inv(T),
        )

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh("../../results/full_mesh_known_poses.ply", mesh)
    o3d.visualization.draw_geometries([mesh])