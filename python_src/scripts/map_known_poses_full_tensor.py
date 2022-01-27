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

    device = o3d.core.Device("CUDA:0")
    tsdf_volume = o3d.t.geometry.TSDFVoxelGrid(
        map_attrs_to_dtypes={'tsdf': o3d.core.float32, 'weight': o3d.core.float32, 'color': o3d.core.float32},
                voxel_size=0.01,
                sdf_trunc=0.05,
                block_resolution=16,
                block_count=10000,
                device=device)

    depth_scale = 5000
    depth_max = 5.0

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)
    intrinsics = o3d.core.Tensor(rgb_camera_intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)

    for _, rgb_cv2, depth_cv2, origin, rot in tqdm(data):
        rgb_o3d = o3d.t.geometry.Image(rgb_cv2).to(device)
        depth_o3d = o3d.t.geometry.Image(depth_cv2).to(device)

        T = np.eye(4)
        T[:3, :3] = o3d.geometry.PointCloud.get_rotation_matrix_from_quaternion(rot)
        T[:3, 3] = origin

        T = np.linalg.inv(T)
        extrinsics = o3d.core.Tensor(T, o3d.core.Dtype.Float64)

        tsdf_volume.integrate(depth_o3d, rgb_o3d, 
                        intrinsics, extrinsics, depth_scale, depth_max)

    mesh = tsdf_volume.extract_surface_mesh()
    o3d.io.write_triangle_mesh("../../results/full_mesh_known_poses.ply", mesh.to_legacy())
    o3d.visualization.draw([mesh.to_legacy()])