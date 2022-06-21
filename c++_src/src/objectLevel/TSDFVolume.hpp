#include <string>

#include "open3d/Open3D.h"

namespace o3d = open3d;

using Model = o3d::t::pipelines::slam::Model;
using Frame = o3d::t::pipelines::slam::Frame;
using Tensor = o3d::core::Tensor;
using Device = o3d::core::Device;

struct FusionModel {
    float class_probability;
    int existence_count_;
    int nonexistence_count_;
    Model model_;
    Frame input_frame_;
    Frame raycast_frame_;
};

class TSDFVolumes {
public:
    TSDFVolumes(float class_score,
                float voxel_size,
                int block_resolution,
                int est_block_count,
                Tensor T_frame_to_model,
                Device device,
                Frame input_frame,
                Frame raycast_frame,
                double depth_scale,
                double depth_max);

    void incrementExistenceCount(int idx) { models_[idx].existence_count_++; }
    void incrementNonExistenceCount(int idx) {
        models_[idx].nonexistence_count_++;
    }

    void addNewInstance(float class_score,
                        float voxel_size,
                        int block_resolution,
                        int est_block_count,
                        Tensor T_frame_to_model,
                        Frame input_frame,
                        Frame raycast_frame);

    void integrateInstance(int frame_id,
                           int inst_id,
                           Frame input_frame,
                           Tensor T_frame_to_Model);
    void updateClassProbability(float detection_score);

    std::vector<float> compute2DIoU(
            const o3d::t::geometry::Image& input_mask) const;

    void updateRaycastFrames(int frame_id, Tensor T_frame_to_model);

private:
    Device device_;
    std::vector<FusionModel> models_;
    double depth_scale_;
    double depth_max_;
};