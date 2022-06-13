#include <string>

#include "open3d/Open3D.h"

namespace o3d = open3d;

using Model = o3d::t::pipelines::slam::Model;
using Frame = o3d::t::pipelines::slam::Frame;
using Tensor = o3d::core::Tensor;
using Device = o3d::core::Device;

class TSDFVolumes {
public:
    TSDFVolumes(float class_score,
               float voxel_size,
               int block_resolution,
               int est_block_count,
               Tensor T_frame_to_model,
               Device device,
               Frame sample_frame,
               double depth_scale,
               double depth_max);
    ~TSDFVolumes();

    void incrementExistenceCount(int idx) { existence_counts_[idx]++; }
    void incrementNonExistenceCount(int idx) { nonexistence_counts_[idx]++; }

    void addNewInstance(float class_score,
                        float voxel_size,
                        int block_resolution,
                        int est_block_count,
                        Tensor T_frame_to_model,
                        Frame sample_frame);

    void integrateInstance(int frame_id,
                           int inst_id,
                           Frame input_frame,
                           Tensor T_frame_to_Model);
    void updateClassProbability(float detection_score);

    std::vector<int> computeMaskOverlap(const o3d::t::geometry::Image& input_mask) const;


private:
    Device device_;
    std::vector<float> class_probabilities_;
    std::vector<int> existence_counts_;
    std::vector<int> nonexistence_counts_;
    std::vector<Model> models_;
    std::vector<Frame> input_frames_;
    std::vector<Frame> raycast_frames_;
    double depth_scale_;
    double depth_max_;
}