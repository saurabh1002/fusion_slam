#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "camera.hpp"
#include "freiburg1_desk.h"
#include "indicators/progress_bar.hpp"
#include "open3d/Open3D.h"

void visualizeModel(o3d::t::pipelines::slam::Model model) {
    auto mesh = model.voxel_grid_.ExtractTriangleMesh();
    mesh.GetVertexNormals();

    auto mesh_legacy = mesh.ToLegacy();

    o3d::visualization::DrawGeometries(
            {std::make_shared<const o3d::geometry::TriangleMesh>(mesh_legacy)});
}

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

int main(int argc, char* argv[]) {
    namespace progress = indicators;
    namespace o3d = open3d;
    using Tensor = o3d::core::Tensor;

    auto argparser = ArgParse(argc, argv);

    auto freiburg_data_path = argparser.get<std::string>("data_root_dir");
    auto config_path = argparser.get<std::string>("config");
    auto n_scans = argparser.get<int>("--n_scans");

    std::ifstream config_file(config_path, std::ios_base::in);
    YAML::Node config = YAML::Load(config_file);

    RGBDCamera rgbd_cam = RGBDCamera(config);

    auto voxel_size = config["TSDF"]["voxel_size"].as<double>();
    auto block_resolution = config["TSDF"]["block_resolution"].as<int>();
    auto est_block_count = config["TSDF"]["est_block_count"].as<int>();

    auto gpu = o3d::core::Device("CUDA:0,1,2,3");
    auto cpu = o3d::core::Device("CPU:0");

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
    auto [_, pose, rgb_t, depth_t] = dataset.At_t(0);

    Tensor T_frame_to_model = Tensor::Eye(4, o3d::core::Float64, cpu);

    auto model = o3d::t::pipelines::slam::Model(voxel_size, block_resolution,
                                                est_block_count,
                                                T_frame_to_model, gpu);

    auto input_frame = o3d::t::pipelines::slam::Frame(
            depth_t.GetRows(), depth_t.GetCols(), rgbd_cam.intrinsics_t_, gpu);
    auto raycast_frame = o3d::t::pipelines::slam::Frame(
            depth_t.GetRows(), depth_t.GetCols(), rgbd_cam.intrinsics_t_, gpu);

    auto start_idx = 200;
    for (std::size_t idx = start_idx; idx < dataset.size(); idx++) {
        bar.set_option(
                progress::option::PostfixText{std::to_string(idx + 1) + "/" +
                                              std::to_string(dataset.size())});
        bar.tick();

        auto [timestamp, _, rgb_t, depth_t] = dataset.At_t(idx);
        auto depth_t_filtered = depth_t.FilterBilateral();

        input_frame.SetDataFromImage("color", rgb_t);
        input_frame.SetDataFromImage("depth", depth_t);

        if (idx > start_idx) {
            auto result = model.TrackFrameToModel(input_frame, raycast_frame,
                                                  rgbd_cam.depth_scale_,
                                                  rgbd_cam.depth_max_, 0.5);
            T_frame_to_model = T_frame_to_model.Matmul(result.transformation_);
        }

        model.UpdateFramePose(idx - start_idx, T_frame_to_model);
        input_frame.SetDataFromImage("depth", depth_t_filtered);
        model.Integrate(input_frame, rgbd_cam.depth_scale_,
                        rgbd_cam.depth_max_);
        model.SynthesizeModelFrame(raycast_frame, rgbd_cam.depth_scale_, 0.1,
                                   rgbd_cam.depth_max_);

        // if (idx % 25 == 0) {
        //     visualizeModel(model);
        // }
    }

    auto mesh = model.voxel_grid_.ExtractTriangleMesh();
    mesh.GetVertexNormals();

    auto mesh_legacy = mesh.ToLegacy();

    o3d::visualization::DrawGeometries(
            {std::make_shared<const o3d::geometry::TriangleMesh>(mesh_legacy)});

    o3d::io::WriteTriangleMeshToPLY("../../../results/mesh_full_ICP_t.ply",
                                    mesh_legacy, false, false, true, true, true,
                                    true);
    return 0;
}