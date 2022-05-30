#include "camera.hpp"

RGBDCamera::RGBDCamera(const YAML::Node& config) {
    auto width = config["RGBCamera"]["width"].as<int>();
    auto height = config["RGBCamera"]["height"].as<int>();

    auto fx = config["RGBCamera"]["fx"].as<double>();
    auto fy = config["RGBCamera"]["fy"].as<double>();
    auto cx = config["RGBCamera"]["cx"].as<double>();
    auto cy = config["RGBCamera"]["cy"].as<double>();

    intrinsics_ = open3d::camera::PinholeCameraIntrinsic(width, height, fx, fy,
                                                         cx, cy);
    intrinsics_t_ = open3d::core::eigen_converter::EigenMatrixToTensor(intrinsics_.intrinsic_matrix_);

    depth_scale_ = config["DepthCamera"]["depth_scale"].as<int>();
    depth_max_ = config["DepthCamera"]["max_range"].as<double>();
}
