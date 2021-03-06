#include <algorithm>
#include <argparse/argparse.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "TSDFVolume.hpp"
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

    auto argparser = ArgParse(argc, argv);

    auto freiburg_data_path = argparser.get<std::string>("data_root_dir");
    auto config_path = argparser.get<std::string>("config");
    auto n_scans = argparser.get<int>("--n_scans");

    std::ifstream config_file(config_path, std::ios_base::in);
    YAML::Node config = YAML::Load(config_file);

    RGBDCamera rgbd_cam = RGBDCamera(config);
    auto depth_scale = rgbd_cam.depth_scale_;
    auto depth_max = rgbd_cam.depth_max_;

    auto voxel_size = config["TSDF"]["voxel_size"].as<double>();
    auto block_resolution = config["TSDF"]["block_resolution"].as<int>();
    auto est_block_count = config["TSDF"]["est_block_count"].as<int>();
    auto score_threshold = config["MaskRCNN"]["score_threshold"].as<float>();

    auto gpu = o3d::core::Device("CPU:0");
    auto cpu = o3d::core::Device("CPU:0");

    auto dataset =
            datasets::freiburg1_desk(freiburg_data_path, config, n_scans);

    auto maskrcnn_data = datasets::maskrcnn(dataset);

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

    o3d::core::Tensor T_frame_to_model =
            o3d::core::Tensor::Eye(4, o3d::core::Float64, cpu);

    auto global_model = o3d::t::pipelines::slam::Model(
            voxel_size, block_resolution, est_block_count, T_frame_to_model,
            gpu);
    std::map<std::string, TSDFVolumes> object_tsdf_dict{};

    auto raycast_mask = o3d::t::pipelines::slam::Frame(
            depth_t.GetRows(), depth_t.GetCols(), rgbd_cam.intrinsics_t_, gpu);
    auto raycast_frame = o3d::t::pipelines::slam::Frame(
            depth_t.GetRows(), depth_t.GetCols(), rgbd_cam.intrinsics_t_, gpu);
    auto input_frame = o3d::t::pipelines::slam::Frame(
            depth_t.GetRows(), depth_t.GetCols(), rgbd_cam.intrinsics_t_, gpu);

    auto start_idx = 200;
    auto end_idx = 300;
    for (std::size_t idx = 0; idx < end_idx - start_idx; idx++) {
        bar.set_option(progress::option::PostfixText{
                std::to_string(idx) + "/" +
                std::to_string(end_idx - start_idx)});
        bar.tick();

        auto [timestamp, _, rgb_t, depth_t] = dataset.At_t(idx + start_idx);
        auto [class_labels, scores, bboxes, masks] =
                maskrcnn_data.At_t(idx + start_idx);

        auto depth_t_filtered = depth_t.FilterBilateral();

        input_frame.SetDataFromImage("color", rgb_t);
        input_frame.SetDataFromImage("depth", depth_t);

        if (idx > 0) {
            auto result = global_model.TrackFrameToModel(
                    input_frame, raycast_frame, rgbd_cam.depth_scale_,
                    rgbd_cam.depth_max_, 0.5);
            T_frame_to_model = T_frame_to_model.Matmul(result.transformation_);
        }
        global_model.UpdateFramePose(idx, T_frame_to_model);
        input_frame.SetDataFromImage("depth", depth_t_filtered);
        global_model.Integrate(input_frame, depth_scale, depth_max);
        global_model.SynthesizeModelFrame(raycast_frame, depth_scale, 0.1,
                                          depth_max);
        // if (!object_tsdf_dict.empty()) {
        //     std::cout << "Debug 5a \n\n" << std::endl;
        //     for (auto& [key, _] : object_tsdf_dict) {
        //         object_tsdf_dict.at(key).updateRaycastFrames(idx,
        //                                                      T_frame_to_model);
        //     }
        // }
        // std::cout << "Debug 6 \n\n" << std::endl;
        if (class_labels.size() > 0) {
            for (size_t i = 0; i < class_labels.size(); i++) {
                const auto& label = class_labels[i];
                const auto score = scores[i];
                input_frame.SetDataFromImage("depth", masks[i]);
                if (score > score_threshold) {
                    if (object_tsdf_dict.find(label) ==
                        object_tsdf_dict.end()) {
                        auto tsdf_volume = TSDFVolumes(
                                idx, score, voxel_size, block_resolution,
                                est_block_count, T_frame_to_model, gpu,
                                input_frame, raycast_mask, depth_scale,
                                depth_max);
                        tsdf_volume.integrateInstance(idx, 0, input_frame,
                                                      T_frame_to_model);
                        object_tsdf_dict.emplace(
                                std::make_pair(label, tsdf_volume));
                    } else {
                        auto iou = object_tsdf_dict.at(label).compute2DIoU(
                                masks[i]);
                        auto argmax_iou = std::distance(
                                iou.cbegin(),
                                std::max_element(iou.cbegin(), iou.cend()));

                        std::cout << iou[argmax_iou] << std::endl;
                        if (iou[argmax_iou] > 0.35) {
                            object_tsdf_dict.at(label).integrateInstance(
                                    idx, argmax_iou, input_frame,
                                    T_frame_to_model);
                        } else {
                            object_tsdf_dict.at(label).addNewInstance(
                                    idx, score, voxel_size, block_resolution,
                                    est_block_count, T_frame_to_model,
                                    input_frame, raycast_mask);
                        }
                    }
                }
            }
        }

        // model.UpdateFramePose(idx - start_idx, T_frame_to_model);
        // input_frame.SetDataFromImage("depth", depth_t_filtered);
        // model.Integrate(input_frame, rgbd_cam.depth_scale_,
        //                 rgbd_cam.depth_max_);
        // model.SynthesizeModelFrame(raycast_frame, depth_scale, 0.1,
        //                            depth_max);

        // for (size_t i = 0; i < class_labels.size(); i++) {
    }

    for (auto& [label, val] : object_tsdf_dict) {
        auto models = val.getModels();
        int inst_id = 0;
        for (auto model : models) {
            auto mesh = model.model_.voxel_grid_.ExtractTriangleMesh();
            mesh.GetVertexNormals();
            auto mesh_legacy = mesh.ToLegacy();
            o3d::visualization::DrawGeometries(
                    {std::make_shared<const o3d::geometry::TriangleMesh>(
                            mesh_legacy)});

        std::string mesh_name = label + "_" + std::to_string(inst_id++) + ".ply";
            o3d::io::WriteTriangleMeshToPLY(
                    "../../../results/object_level_icp/" + mesh_name, mesh_legacy, false,
                    false, true, true, true, true);
        }
    }
    return 0;
}

// auto mesh = model.voxel_grid_.ExtractTriangleMesh();
// mesh.GetVertexNormals();

// auto mesh_legacy = mesh.ToLegacy();

// o3d::visualization::DrawGeometries(
//         {std::make_shared<const o3d::geometry::TriangleMesh>(mesh_legacy)});

// o3d::io::WriteTriangleMeshToPLY("../../../results/mesh_full_ICP_t.ply",
//                                 mesh_legacy, false, false, true, true, true,
//                                 true);