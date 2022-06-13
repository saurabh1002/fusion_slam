#include "freiburg1_desk.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "open3d/Open3D.h"

namespace fs = std::filesystem;
namespace o3d = open3d;

namespace {
std::tuple<std::vector<double>,
           std::vector<Eigen::Vector<double, 7>>,
           std::vector<std::string>,
           std::vector<std::string>>
readData(const fs::path& dataset_path, int n_scans) {
    auto associations_file = dataset_path / "associations_rgbd_gt.txt";

    std::vector<double> timestamps;
    std::vector<Eigen::Vector<double, 7>> poses;
    std::vector<std::string> rgb_files;
    std::vector<std::string> depth_files;

    std::string rgb_file;
    std::string depth_file;
    float tx, ty, tz, qx, qy, qz, qw;
    double timestamp;
    float _;

    std::ifstream data_in(associations_file, std::ios_base::in);
    // clang-format off
    if(n_scans == -1){
    while(data_in >> timestamp >> rgb_file >>
                     _ >> depth_file >>
                     _ >> tx >> ty >> tz >>
                     qx >> qy >> qz >> qw) {
            // clang-format on
            timestamps.emplace_back(timestamp);
            rgb_files.emplace_back(dataset_path / rgb_file);
            depth_files.emplace_back(dataset_path / depth_file);
            Eigen::Vector<double, 7> pose{tx, ty, tz, qx, qy, qz, qw};
            poses.emplace_back(pose);
        }
    } else {
        for (int i = 0; i < n_scans; i++) {
            // clang-format off
            data_in >> timestamp >> rgb_file >>
                       _ >> depth_file >>
                       _ >> tx >> ty >> tz >>
                       qx >> qy >> qz >> qw;
            // clang-format on
            timestamps.emplace_back(timestamp);
            rgb_files.emplace_back(dataset_path / rgb_file);
            depth_files.emplace_back(dataset_path / depth_file);
            Eigen::Vector<double, 7> pose{tx, ty, tz, qx, qy, qz, qw};
            poses.emplace_back(pose);
        }
    }
    return std::make_tuple(timestamps, poses, rgb_files, depth_files);
}

}  // namespace

namespace datasets {
using Image = o3d::geometry::Image;
using Image_t = o3d::t::geometry::Image;
auto gpu = o3d::core::Device("CUDA:0,1,2,3");

freiburg1_desk::freiburg1_desk(const std::string& data_root_dir,
                               const YAML::Node& cfg,
                               int n_scans)
    : cfg_(cfg) {
    auto freiburg_root_dir_ = fs::absolute(fs::path(data_root_dir));

    std::tie(time_, poses_, rgb_files_, depth_files_) =
            readData(freiburg_root_dir_, n_scans);
}

std::tuple<double, Eigen::Vector<double, 7>, Image, Image> freiburg1_desk::At(
        int idx) const {
    double timestamp = time_[idx];
    Eigen::Vector<double, 7> pose = poses_[idx];
    Image rgb_image_8bit;
    Image depth_image_16bit;
    if (open3d::io::ReadImage(rgb_files_[idx], rgb_image_8bit) &&
        open3d::io::ReadImage(depth_files_[idx], depth_image_16bit)) {
        return std::make_tuple(timestamp, pose, rgb_image_8bit,
                               depth_image_16bit);
    }
    o3d::utility::LogWarning("Failed to read following files:\n{}\n{}",
                             rgb_files_[idx], depth_files_[idx]);
    exit(1);
}

std::tuple<double, Eigen::Vector<double, 7>, Image_t, Image_t>
freiburg1_desk::At_t(int idx) const {
    double timestamp = time_[idx];
    Eigen::Vector<double, 7> pose = poses_[idx];
    Image rgb_image_8bit;
    Image depth_image_16bit;
    if (open3d::io::ReadImage(rgb_files_[idx], rgb_image_8bit) &&
        open3d::io::ReadImage(depth_files_[idx], depth_image_16bit)) {
        auto rgb_t = o3d::t::geometry::Image::FromLegacy(rgb_image_8bit, gpu);
        auto depth_t =
                o3d::t::geometry::Image::FromLegacy(depth_image_16bit, gpu);
        return std::make_tuple(timestamp, pose, rgb_t, depth_t);
    }
    o3d::utility::LogWarning("Failed to read following files:\n{}\n{}",
                             rgb_files_[idx], depth_files_[idx]);
    exit(1);
}

maskrcnn::maskrcnn(freiburg1_desk data_fr1) : data_fr1_(std::move(data_fr1)) {}
std::tuple<std::vector<std::string>,
           std::vector<float>,
           std::vector<std::vector<int>>,
           std::vector<Image>>
maskrcnn::At(int idx) const {
    const auto& masks_path = fs::path(data_fr1_.rgb_files_[idx]).parent_path() /
                             fs::path(data_fr1_.rgb_files_[idx]).stem();

    std::string class_label;
    std::vector<std::string> class_labels;
    std::ifstream labels_file(masks_path / "class_labels.txt",
                              std::ios_base::in);
    while (std::getline(labels_file, class_label)) {
        class_labels.emplace_back(class_label);
    }

    float score;
    std::vector<float> class_scores;
    std::ifstream scores_file(masks_path / "scores.txt", std::ios_base::in);
    while (scores_file >> score) {
        class_scores.emplace_back(score);
    }

    double tlx, tly, brx, bry;
    std::vector<std::vector<int>> bboxes;
    std::ifstream bboxes_file(masks_path / "bboxes.txt", std::ios_base::in);
    while (bboxes_file >> tlx >> tly >> brx >> bry) {
        bboxes.emplace_back(
                std::vector<int>{static_cast<int>(tlx), static_cast<int>(tly),
                                 static_cast<int>(brx), static_cast<int>(bry)});
    }

    std::vector<Image> masks;
    for (size_t i = 0; i < class_labels.size(); i++) {
        const auto& image =
                masks_path.string() + "/masks/" + std::to_string(i) + ".png";
        Image mask;
        o3d::io::ReadImage(image, mask);
        masks.emplace_back(mask);
    }
    return std::make_tuple(class_labels, class_scores, bboxes, masks);
}

std::tuple<std::vector<std::string>,
           std::vector<float>,
           std::vector<std::vector<int>>,
           std::vector<Image_t>>
maskrcnn::At_t(int idx) const {
    const auto& masks_path = fs::path(data_fr1_.rgb_files_[idx]).parent_path() /
                             fs::path(data_fr1_.rgb_files_[idx]).stem();

    std::string class_label;
    std::vector<std::string> class_labels;
    std::ifstream labels_file(masks_path / "class_labels.txt",
                              std::ios_base::in);
    while (std::getline(labels_file, class_label)) {
        class_labels.emplace_back(class_label);
    }

    float score;
    std::vector<float> class_scores;
    std::ifstream scores_file(masks_path / "scores.txt", std::ios_base::in);
    while (scores_file >> score) {
        class_scores.emplace_back(score);
    }

    double tlx, tly, brx, bry;
    std::vector<std::vector<int>> bboxes;
    std::ifstream bboxes_file(masks_path / "bboxes.txt", std::ios_base::in);
    while (bboxes_file >> tlx >> tly >> brx >> bry) {
        bboxes.emplace_back(
                std::vector<int>{static_cast<int>(tlx), static_cast<int>(tly),
                                 static_cast<int>(brx), static_cast<int>(bry)});
    }

    std::vector<Image_t> masks;
    for (size_t i = 0; i < class_labels.size(); i++) {
        const auto& image =
                masks_path.string() + "/masks/" + std::to_string(i) + ".png";
        Image mask;
        o3d::io::ReadImage(image, mask);
        masks.emplace_back(o3d::t::geometry::Image::FromLegacy(mask, gpu));
    }
    return std::make_tuple(class_labels, class_scores, bboxes, masks);
}
}  // namespace datasets