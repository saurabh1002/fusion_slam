#!/usr/bin/env python

# ==============================================================================

# @Authors: Saurabh Gupta
# @email: s7sagupt@uni-bonn.de

# ==============================================================================

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
from pyexpat import model
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
    data = DatasetRGBD(datadir)

    rgb_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
    rgb_camera_intrinsic.set_intrinsics(640, 480, 517.3, 516.5, 318.6, 255.3)
    intrinsics = o3d.core.Tensor(rgb_camera_intrinsic.intrinsic_matrix, o3d.core.Dtype.Float64)
    depth_scale = 5000
    depth_max = 3.0

    device = o3d.core.Device("CUDA:0")
    tsdf_volume = o3d.t.geometry.TSDFVoxelGrid(
        map_attrs_to_dtypes={'tsdf': o3d.core.float32, 'weight': o3d.core.float32, 'color': o3d.core.float32},
                voxel_size=0.01,
                sdf_trunc=0.03,
                block_resolution=10,
                block_count=40000,
                device=device)

    I = o3d.core.Tensor(np.eye(4), o3d.core.Dtype.Float64)
    extrinsics = o3d.core.Tensor(np.eye(4), o3d.core.Dtype.Float64)

    t, rgb_cv2, depth_cv2 = data[0]
    rgb_o3d = o3d.t.geometry.Image(rgb_cv2).to(device)
    depth_o3d = o3d.t.geometry.Image(depth_cv2).to(device)

    tsdf_volume.integrate(depth_o3d, rgb_o3d, 
                        intrinsics, extrinsics, depth_scale, depth_max)

    poses = []
    for i, [t, rgb_cv2, depth_cv2] in tqdm(enumerate(data)):
        if i % 2 != 0:
            continue
        rgb_o3d = o3d.t.geometry.Image(rgb_cv2).to(device)
        depth_o3d = o3d.t.geometry.Image(depth_cv2).to(device)

        model_raycast = tsdf_volume.raycast(intrinsics, extrinsics.inv(),
                                640, 480, depth_scale,
                                weight_threshold=0.2, 
                                raycast_result_mask=2
                            )
        model_depth = o3d.t.geometry.Image(model_raycast[2])

        # o3d.visualization.draw_geometries([model_depth.to_legacy()])

        model_pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            model_depth, intrinsics, I, depth_scale, depth_max
        )
        model_pcd.estimate_normals()

        frame_pcd = o3d.t.geometry.PointCloud.create_from_depth_image(
            depth_o3d, intrinsics, I, depth_scale, depth_max
        )

        result = o3d.t.pipelines.registration.icp(
            frame_pcd,
            model_pcd,
            0.1,
            I,
            o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        # if i % 25 == 0:
        #     mesh = tsdf_volume.extract_surface_mesh()
        #     o3d.visualization.draw_geometries([mesh.to_legacy()])

        extrinsics = extrinsics.matmul(result.transformation)
        tsdf_volume.integrate(depth_o3d, rgb_o3d, 
                        intrinsics, extrinsics.inv(), depth_scale, depth_max)

        poses.append(odom_from_SE3(t, extrinsics.numpy()))

    np.savetxt("../../results/poses_model_to_frame.txt", np.array(poses))
    mesh = tsdf_volume.extract_surface_mesh()
    o3d.io.write_triangle_mesh("../../results/full_mesh_model_to_frame.ply", mesh.to_legacy())
    o3d.visualization.draw_geometries([mesh.to_legacy()])
