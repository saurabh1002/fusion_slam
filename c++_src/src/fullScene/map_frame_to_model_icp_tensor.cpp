#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "freiburg1_desk.h"
#include "indicators/progress_bar.hpp"
#include "open3d/Open3D.h"

argparse::ArgumentParser ArgParse(int argc, char* argv[]) {
    argparse::ArgumentParser argparser("Mapping with known poses pipeline");
    argparser.add_argument("data_root_dir")
            .help("The full path to the dataset");
    argparser.add_argument("config").help(
            "The full path to the yaml config file");
    argparser.add_argument("--n_scans")
            .help("How many scans to map")
            .default_value(int(-1))
            .action([](const std::string& value) { return std::stoi(value); });

    try {
        argparser.parse_args(argc, argv);
    } catch (const std::runtime_error& err) {
        std::cerr << "Invalid Arguments " << std::endl;
        std::cerr << err.what() << std::endl;
        std::cerr << argparser;
        exit(0);
    }
    auto data_root_dir = argparser.get<std::string>("data_root_dir");
    if (!std::filesystem::exists(data_root_dir)) {
        std::cerr << data_root_dir << "path doesn't exists" << std::endl;
        exit(1);
    }
    auto config = argparser.get<std::string>("config");
    if (!std::filesystem::exists(config)) {
        std::cerr << config << "file doesn't exists" << std::endl;
        exit(1);
    }
    return argparser;
}

Eigen::Matrix4d TF_from_poses(const Eigen::Vector<double, 7>& pose) {
    Eigen::Matrix4d extrinsics;
    extrinsics.setIdentity();

    auto R = open3d::geometry::PointCloud::GetRotationMatrixFromQuaternion(
            Eigen::Vector4d{pose[6], pose[3], pose[4], pose[5]});
    extrinsics.block<3, 3>(0, 0) = R;
    extrinsics.block<3, 1>(0, 3) = Eigen::Vector3d{pose[0], pose[1], pose[2]};

    return extrinsics;
}

int main(int argc, char* argv[]) {
    namespace progress = indicators;
    namespace o3d = open3d;

    auto argparser = ArgParse(argc, argv);

    auto freiburg_data_path = argparser.get<std::string>("data_root_dir");
    auto config_path = argparser.get<std::string>("config");
    auto n_scans = argparser.get<int>("--n_scans");

    std::ifstream config_file(config_path, std::ios_base::in);
    YAML::Node config = YAML::Load(config_file);

    RGBDCamera rgbd_cam = RGBDCamera(config);

    auto voxel_size = config["TSDF"]["voxel_size"].as<double>();
    auto sdf_trunc = config["TSDF"]["sdf_trunc"].as<double>();
    auto space_carving = config["TSDF"]["space_carving"].as<bool>();

    auto device = o3d::core::Device("CPU:0");

    auto dataset =
            datasets::freiburg1_desk(freiburg_data_path, config, n_scans);

    // clang-format off
    progress::ProgressBar bar{
        progress::option::BarWidth{40},
        progress::option::Start{"Integrating TSDF: ["},
        progress::option::Fill{"="},
        progress::option::Lead{">"},
        progress::option::Remainder{"."},
        progress::option::End{"]"},
        progress::option::MaxProgress{dataset.size()},
        progress::option::ForegroundColor{progress::Color::green},
        progress::option::FontStyles{std::vector<progress::FontStyle>{progress::FontStyle::bold}}
    };
    // clang-format on
    const o3d::core::Tensor& I =
            o3d::core::Tensor::Eye(4, o3d::core::Float64, device);

    auto voxel_block_grid = o3d::t::geometry::VoxelBlockGrid(
            {"tsdf", "weight", "color"},
            {o3d::core::Dtype::Float32, o3d::core::Dtype::Float32,
             o3d::core::Dtype::Float32},
            {{1}, {1}, {3}}, voxel_size, 16, 10000, device);

    auto extrinsics_t =
            o3d::core::Tensor::Eye(4, o3d::core::Dtype::Float64, device);

    auto [timestamp, pose, rgbImage, depthImage] = dataset[1];
    auto rgb_t = o3d::t::geometry::Image::FromLegacy(rgbImage, device);
    auto depth_t = o3d::t::geometry::Image::FromLegacy(depthImage, device);

    o3d::core::Tensor block_coords = voxel_block_grid.GetUniqueBlockCoordinates(
            depth_t, rgbd_cam.intrinsics_t_, extrinsics_t,
            rgbd_cam.depth_scale_, rgbd_cam.depth_max_);

    voxel_block_grid.Integrate(block_coords, depth_t, rgb_t,
                               rgbd_cam.intrinsics_t_, extrinsics_t,
                               rgbd_cam.depth_scale_, rgbd_cam.depth_max_);

    double max_correspondance_distance = 0.1;
    for (std::size_t idx = 1; idx < dataset.size(); idx++) {
        bar.set_option(
                progress::option::PostfixText{std::to_string(idx + 1) + "/" +
                                              std::to_string(dataset.size())});
        bar.tick();
        
        auto [timestamp, _, rgbImage, depthImage] = dataset[idx];
        rgb_t = o3d::t::geometry::Image::FromLegacy(rgbImage, device);
        depth_t = o3d::t::geometry::Image::FromLegacy(depthImage, device);

        block_coords = voxel_block_grid.GetUniqueBlockCoordinates(
            depth_t, rgbd_cam.intrinsics_t_, extrinsics_t,
            rgbd_cam.depth_scale_, rgbd_cam.depth_max_);

        auto raycast_res = voxel_block_grid.RayCast(
                    block_coords, rgbd_cam.intrinsics_t_, extrinsics_t,
                    rgbd_cam.intrinsics_.width_, rgbd_cam.intrinsics_.height_, {"depth", "color"},
                    rgbd_cam.depth_scale_, 0.1, rgbd_cam.depth_max_, std::min(idx * 1.0f, 3.0f));

        o3d::t::geometry::Image depth_raycast(raycast_res["depth"]);
        auto model_pcd = o3d::t::geometry::PointCloud::CreateFromDepthImage(
                depth_raycast, rgbd_cam.intrinsics_t_, I,
                               rgbd_cam.depth_scale_, rgbd_cam.depth_max_ );
        // model_pcd.EstimateNormals();

        auto frame_pcd = o3d::t::geometry::PointCloud::CreateFromDepthImage(
                depth_t, rgbd_cam.intrinsics_t_, I,
                               rgbd_cam.depth_scale_, rgbd_cam.depth_max_);

        auto result = o3d::t::pipelines::registration::ICP(
                frame_pcd, model_pcd, max_correspondance_distance,
                I, o3d::t::pipelines::registration::TransformationEstimationPointToPoint(),
                o3d::t::pipelines::registration::ICPConvergenceCriteria(),
                2 * voxel_size);

        // Works fine
        extrinsics_t = (result.transformation_.Inverse().Contiguous()).Matmul(extrinsics_t);
        // extrinsics_t = extrinsics_t.Matmul(result.transformation_.Inverse().Contiguous());
        voxel_block_grid.Integrate(block_coords, depth_t, rgb_t,
                               rgbd_cam.intrinsics_t_, extrinsics_t,
                               rgbd_cam.depth_scale_, rgbd_cam.depth_max_);
        
        if (idx % 25 == 0){
            o3d::t::geometry::Image color_raycast(raycast_res["color"]);
            o3d::visualization::DrawGeometries(
                    {std::make_shared<o3d::geometry::Image>(
                            color_raycast.ToLegacy())});

            o3d::visualization::DrawGeometries({std::make_shared<const o3d::geometry::Image>(
                                depth_raycast.ColorizeDepth(rgbd_cam.depth_scale_, 0.1,
                                                        rgbd_cam.depth_max_).ToLegacy())});
            auto mesh = voxel_block_grid.ExtractTriangleMesh();
            mesh.GetVertexNormals();

            auto mesh_legacy = mesh.ToLegacy();

            o3d::visualization::DrawGeometries({
                            std::make_shared<const o3d::geometry::TriangleMesh>(
                                    mesh_legacy)});
        }
    }

    auto mesh = voxel_block_grid.ExtractTriangleMesh();
    mesh.GetVertexNormals();

    auto mesh_legacy = mesh.ToLegacy();

    o3d::visualization::DrawGeometries({std::make_shared<const o3d::geometry::TriangleMesh>(
                            mesh_legacy)});

    o3d::io::WriteTriangleMeshToPLY("../results/mesh_model_to_frame_full.ply",
                                    mesh_legacy, false, false, true, true, true,
                                    true);
    return 0;
}