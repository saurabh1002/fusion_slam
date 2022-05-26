#pragma once

#include "open3d/Open3D.h"
#include "yaml-cpp/yaml.h"

class RGBDCamera {
public:
    explicit RGBDCamera(const YAML::Node& config);

public:
    open3d::camera::PinholeCameraIntrinsic intrinsics_;
    open3d::core::Tensor intrinsics_t_;
    double depth_scale_;
    double depth_max_;
};