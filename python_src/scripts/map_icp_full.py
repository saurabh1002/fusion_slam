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
from utils.dataloader import DatasetPCL

import open3d as o3d


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100),
    )
    return pcd_down, pcd_fpfh


def execute_global_registration(
    source_down, target_down, source_fpfh, target_fpfh, voxel_size
):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


def refine_registration(source, target, result_ransac, voxel_size):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        distance_threshold,
        result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result


def odom_from_SE3(TF: np.ndarray) -> (list):
    """
    Arguments
    ---------
    - translation: [tx, ty, tz] 3d translation wrt reference frame
    - quaternion: [qw, qx, qy, qz] quaternion representing rotation wrt reference frame

    Returns
    -------
    SE3 transformation matrix combining the rotation and translation
    """
    origin = TF[:-1, -1]
    rot_quat = tf.Rotation.from_matrix(TF[:3, :3]).as_quat()
    return list(np.r_[origin, rot_quat])


if __name__ == "__main__":

    datadir = "../../datasets/rgbd_dataset_freiburg1_desk/"

    PCL_data = DatasetPCL(datadir)

    merged_pcd = o3d.geometry.PointCloud()

    I = np.eye(4)
    T = np.eye(4)
    loss = o3d.pipelines.registration.TukeyLoss(k=0.1)

    poses = []

    voxel_size = 0.1
    # max_corr_dist = 0.5
    skip_frames = 1
    for n in tqdm(range(0, len(PCL_data) - skip_frames, skip_frames)):
        # o3d.visualization.draw_geometries([merged_pcd], "TUM desk voxel map")
        # if n >= skip_frames:
        #     break
        target = PCL_data[n][1]
        target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
        target.estimate_normals()

        source = PCL_data[n + skip_frames][1]
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
        source.estimate_normals()

        glob_result = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size
        )

        # o3d.visualization.draw_geometries(
        #     [target_down, source_down.transform(glob_result.transformation)],
        #     "TUM desk voxel map",
        # )

        local_result = refine_registration(source, target, glob_result, voxel_size)

        # o3d.visualization.draw_geometries(
        #     [target, source.transform(local_result.transformation)],
        #     "TUM desk voxel map",
        # )
        # reg_result = o3d.pipelines.registration.registration_icp(
        #     source,
        #     target,
        #     max_corr_dist,
        #     I,
        #     o3d.pipelines.registration.TransformationEstimationPointToPlane(loss),
        #     o3d.pipelines.registration.ICPConvergenceCriteria(
        #         relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100
        #     ),
        # )

        T = np.array(local_result.transformation) @ T

        poses.append([PCL_data[n + skip_frames][0]] + odom_from_SE3(T))

        merged_pcd += source.transform(T)

        # Downsample poincloud occassionally to reduce redundant points and computation cost
        # if n % 5 == 0:
        merged_pcd = merged_pcd.voxel_down_sample(0.001)

    np.savetxt("../../results/poses_icp.txt", np.array(poses))
    voxel_map = o3d.geometry.VoxelGrid.create_from_point_cloud(
        merged_pcd, voxel_size=0.01
    )
    # o3d.visualization.draw_geometries([voxel_map], "TUM desk voxel map")
    # o3d.io.write_point_cloud(
    #     datadir + "results/pcl_icp_map.xyzrgb", merged_pcd, print_progress=True
    # )
