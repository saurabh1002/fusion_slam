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
    : device_(device), depth_scale_(depth_scale), depth_max_(depth_max) {
    auto tmp_model = Model(voxel_size, block_resolution, est_block_count,
                           T_frame_to_model, device_);
    models_ = {FusionModel{class_score, 2, 1, tmp_model, sample_frame,
                            sample_frame}};
}

void TSDFVolumes::addNewInstance(float class_score,
                                 float voxel_size,
                                 int block_resolution,
                                 int est_block_count,
                                 Tensor T_frame_to_model,
                                 Frame sample_frame) {
    auto tmp_model = Model(voxel_size, block_resolution, est_block_count,
                           T_frame_to_model, device_);
    models_.emplace_back(FusionModel{class_score, 2, 1, std::move(tmp_model),
                                     sample_frame, sample_frame});
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
    auto mask_tensor = input_mask.AsTensor();
    for (auto& model : models_) {
        auto raycast_tensor = model.raycast_frame_.GetData("depth_map");
        auto frame_intersection =
                ((raycast_tensor.Gt(0)) && (mask_tensor.Gt(0))).Sum({0, 1});
        auto frame_union =
                ((raycast_tensor.Gt(0)) || (mask_tensor.Gt(0))).Sum({0, 1});
        iou.emplace_back(frame_intersection.ToFlatVector<float>()[0] /
                         frame_union.ToFlatVector<float>()[0]);
    }
    return iou;
}

void TSDFVolumes::updateRaycastFrames(int frame_id, Tensor T_frame_to_model){
    for (auto& model : models_){
        model.model_.UpdateFramePose(frame_id, T_frame_to_model);
        model.model_.SynthesizeModelFrame(model.raycast_frame_, depth_scale_, 0.1, depth_max_);
    }
}