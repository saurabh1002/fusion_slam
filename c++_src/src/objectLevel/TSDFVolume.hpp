#include <string>

#include "open3d/Open3D.h"

namespace o3d = open3d;

auto gpu = o3d::core::Device("CUDA:0");
auto cpu = o3d::core::Device("CPU:0");

class TSDFVolume {
public:
    TSDFVolume(std::string class_name,
               float class_score,
               float voxel_size,
               int block_resolution,
               int est_block_count,
               o3d::core::Tensor T_frame_to_model,
               o3d::core::device device,
               o3d::t::pipelines::slam::Frame sample_frame);
    ~TSDFVolume();

    void incrementExistenceCount() { existence_count_++; }
    void incrementNonExistenceCount() { nonexistence_count_++; }
    void updateClassProbability(float detection_score);

private:
    std::string class_name_;
    float class_probability_;
    int existence_count_ = 1;
    int nonexistence_count_ = 1;

    o3d::t::pipelines::slam::Model model_;
    o3d::t::pipelines::slam::Frame input_frame_;
    o3d::t::pipelines::slam::Frame raycast_frame_;
}