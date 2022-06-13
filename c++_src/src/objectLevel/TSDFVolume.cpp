#include "TSDFVolume.hpp"

#include <string>

#include "open3d/Open3D.h"

TSDFVolumes::TSDFVolumes(float class_score,
                         float voxel_size,
                         int block_resolution,
                         int est_block_count,
                         Tensor T_frame_to_model,
                         Device device,
                         Frame sample_frame,
                         double depth_scale,
                         double depth_max)
    : device_(device),
      class_probabilities_({class_score}),
      models_({Model(voxel_size,
                     block_resolution,
                     est_block_count,
                     T_frame_to_model,
                     device_)}),
      input_frames_({sample_frame}),
      raycast_frames_({sample_frame}),
      existence_counts_({2}),
      nonexistence_counts_({1}),
      depth_scale_(depth_scale),
      depth_max_(depth_max) {}

void TSDFVolumes::addNewInstance(float class_score,
                                 float voxel_size,
                                 int block_resolution,
                                 int est_block_count,
                                 Tensor T_frame_to_model,
                                 Frame sample_frame) {
    class_probabilities_.emplace_back(class_score);
    models_.emplace_back(Model(voxel_size, block_resolution, est_block_count,
                               T_frame_to_model, device_));
    input_frames_.emplace_back(sample_frame);
    raycast_frames_.emplace_back(sample_frame);
    existence_counts_.emplace_back(2);
    nonexistence_counts_.emplace_back(1);
}

void TSDFVolumes::integrateInstance(int frame_id,
                                    int inst_id,
                                    Frame input_frame,
                                    Tensor T_frame_to_Model) {
    models_[inst_id].UpdateFramePose(frame_id, T_frame_to_Model);
    models_[inst_id].Integrate(input_frame, depth_scale_, depth_max_);
    models_[inst_id].SynthesizeModelFrame(raycast_frames_[inst_id],
                                          depth_scale_, 0.1, depth_max_);
    incrementExistenceCount(inst_id);
}

void TSDFVolumes::updateClassProbability(float detection_score) {}

std::vector<int> TSDFVolumes::computeMaskOverlap(
        const o3d::t::geometry::Image& input_mask) const {
    auto mask_tensor = input_mask.AsTensor();
    for (auto& model : models_) {
      
    }
}