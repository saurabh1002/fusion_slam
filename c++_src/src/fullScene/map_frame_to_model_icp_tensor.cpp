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
    // auto sdf_trunc = config["TSDF"]["sdf_trunc"].as<double>();
    // auto space_carving = config["TSDF"]["space_carving"].as<bool>();

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
    auto [_, pose, rgbImage, depthImage] = dataset[0];
    auto rgb_t = o3d::t::geometry::Image::FromLegacy(rgbImage, device);
    auto depth_t = o3d::t::geometry::Image::FromLegacy(depthImage, device);

    o3d::core::Tensor T_frame_to_model =
            o3d::core::Tensor::Eye(4, o3d::core::Float64, device);

    auto model = o3d::t::pipelines::slam::Model(voxel_size, 16, 40000,
                                                T_frame_to_model, device);

    auto input_frame =
            o3d::t::pipelines::slam::Frame(depth_t.GetRows(), depth_t.GetCols(),
                                           rgbd_cam.intrinsics_t_, device);
    auto raycast_frame =
            o3d::t::pipelines::slam::Frame(depth_t.GetRows(), depth_t.GetCols(),
                                           rgbd_cam.intrinsics_t_, device);

    auto start_idx = 200;
    for (std::size_t idx = start_idx; idx < dataset.size(); idx++) {
        bar.set_option(
                progress::option::PostfixText{std::to_string(idx + 1) + "/" +
                                              std::to_string(dataset.size())});
        bar.tick();

        auto [timestamp, _, rgbImage, depthImage] = dataset[idx];
        rgb_t = o3d::t::geometry::Image::FromLegacy(rgbImage, device);
        depth_t = o3d::t::geometry::Image::FromLegacy(depthImage, device);
        // depth_t = depth_t.FilterBilateral();

        input_frame.SetDataFromImage("color", rgb_t);
        input_frame.SetDataFromImage("depth", depth_t);

        if (idx > start_idx) {
            auto result = model.TrackFrameToModel(input_frame, raycast_frame,
                                                  rgbd_cam.depth_scale_,
                                                  rgbd_cam.depth_max_, 0.5);
            T_frame_to_model = T_frame_to_model.Matmul(result.transformation_);
        }

        model.UpdateFramePose(idx - start_idx, T_frame_to_model);
        model.Integrate(input_frame, rgbd_cam.depth_scale_,
                        rgbd_cam.depth_max_);
        model.SynthesizeModelFrame(raycast_frame, rgbd_cam.depth_scale_, 0.1,
                                   rgbd_cam.depth_max_);

        if (idx % 25 == 0) {
            auto mesh = model.voxel_grid_.ExtractTriangleMesh();
            mesh.GetVertexNormals();

            auto mesh_legacy = mesh.ToLegacy();

            o3d::visualization::DrawGeometries(
                    {std::make_shared<const o3d::geometry::TriangleMesh>(
                            mesh_legacy)});
        }
    }

    auto mesh = model.voxel_grid_.ExtractTriangleMesh();
    mesh.GetVertexNormals();

    auto mesh_legacy = mesh.ToLegacy();

    o3d::visualization::DrawGeometries(
            {std::make_shared<const o3d::geometry::TriangleMesh>(mesh_legacy)});

    o3d::io::WriteTriangleMeshToPLY("../results/mesh_model_to_frame_full.ply",
                                    mesh_legacy, false, false, true, true, true,
                                    true);
    return 0;
}