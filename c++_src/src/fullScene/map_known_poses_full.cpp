#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "freiburg1_desk.h"
#include "indicators/progress_bar.hpp"
#include "open3d/Open3D.h"
#include "yaml-cpp/yaml.h"

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
    // auto space_carving = config["TSDF"]["space_carving"].as<bool>();

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

    auto tsdf_volume = o3d::pipelines::integration::ScalableTSDFVolume(
            voxel_size, sdf_trunc,
            o3d::pipelines::integration::TSDFVolumeColorType::RGB8);

    for (std::size_t idx = 0; idx < dataset.size(); idx++) {
        bar.set_option(
                progress::option::PostfixText{std::to_string(idx + 1) + "/" +
                                              std::to_string(dataset.size())});
        bar.tick();
        auto [timestamp, pose, rgbImage, depthImage] = dataset.At(idx);
        auto rgbdImage = o3d::geometry::RGBDImage::CreateFromColorAndDepth(
                rgbImage, depthImage, rgbd_cam.depth_scale_,
                rgbd_cam.depth_max_, false);

        auto extrinsics = TF_from_poses(pose);
        tsdf_volume.Integrate(*rgbdImage, rgbd_cam.intrinsics_,
                              extrinsics.inverse());
    }

    auto mesh = tsdf_volume.ExtractTriangleMesh();
    mesh->ComputeVertexNormals();
    o3d::io::WriteTriangleMeshToPLY("../results/mesh_known_poses_full.ply",
                                    *mesh, false, false, true, true, true,
                                    true);
    o3d::visualization::DrawGeometries(
            std::vector<std::shared_ptr<const o3d::geometry::Geometry>>{mesh});
    return 0;
}