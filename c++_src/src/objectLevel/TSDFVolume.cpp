#include "TSDFVolume.hpp"

#include <algorithm>
#include <string>

#include "open3d/Open3D.h"

TSDFVolumes::TSDFVolumes(float class_score,
                         float voxel_size,
                         int block_resolution,
                         int est_block_count,
                         Tensor T_frame_to_model,
                         Device device,
                         Frame input_frame,
                         Frame raycast_frame,
                         double depth_scale,
                         double depth_max)
    : device_(device), depth_scale_(depth_scale), depth_max_(depth_max) {
    auto tmp_model = Model(voxel_size, block_resolution, est_block_count,
                           T_frame_to_model, device_);
    models_ = {FusionModel{class_score, 2, 1, tmp_model, input_frame,
                           raycast_frame}};
}

void TSDFVolumes::addNewInstance(float class_score,
                                 float voxel_size,
                                 int block_resolution,
                                 int est_block_count,
                                 Tensor T_frame_to_model,
                                 Frame input_frame,
                                 Frame raycast_frame) {
    auto tmp_model = Model(voxel_size, block_resolution, est_block_count,
                           T_frame_to_model, device_);
    models_.emplace_back(FusionModel{class_score, 2, 1, std::move(tmp_model),
                                     input_frame, raycast_frame});
}

void TSDFVolumes::integrateInstance(int frame_id,
                                    int inst_id,
                                    Frame input_frame,
                                    Tensor T_frame_to_Model) {
    models_[inst_id].model_.UpdateFramePose(frame_id, T_frame_to_Model);
    models_[inst_id].model_.Integrate(input_frame, depth_scale_, depth_max_);
    models_[inst_id].model_.SynthesizeModelFrame(
            models_[inst_id].raycast_frame_, depth_scale_, 0.1, depth_max_);
    incrementExistenceCount(inst_id);
}

void TSDFVolumes::updateClassProbability(float detection_score) {}

std::vector<float> TSDFVolumes::compute2DIoU(
        const o3d::t::geometry::Image& input_mask) const {
    std::vector<float> iou;
    iou.reserve(models_.size());
    auto mask_tensor = input_mask.AsTensor().ToFlatVector<uint16_t>();
    for (auto& model : models_) {
        auto raycast_tensor =
                model.raycast_frame_.GetData("depth").ToFlatVector<float>();
        std::vector<int> intersection_;
        std::vector<int> union_;
        std::transform(mask_tensor.cbegin(), mask_tensor.cend(),
                       raycast_tensor.cbegin(), intersection_.begin(),
                       [](auto a, auto b) {
                           if (a > 0.0 && b > 0.0) {
                               return 1;
                           } else {
                               return 0;
                           }
                       });
        std::transform(mask_tensor.cbegin(), mask_tensor.cend(),
                       raycast_tensor.cbegin(), union_.begin(),
                       [](float a, float b) {
                           if (a > 0.0 || b > 0.0) {
                               return 1;
                           } else {
                               return 0;
                           }
                       });

        iou.emplace_back(std::accumulate(intersection_.cbegin(),
                                         intersection_.cend(), 0.0) /
                         std::accumulate(union_.cbegin(), union_.cend(), 0.0));
    }
    return iou;
}

void TSDFVolumes::updateRaycastFrames(int frame_id, Tensor T_frame_to_model) {
    for (auto& model : models_) {
        model.model_.UpdateFramePose(frame_id, T_frame_to_model);
        model.model_.SynthesizeModelFrame(model.raycast_frame_, depth_scale_,
                                          0.1, depth_max_);
    }
}