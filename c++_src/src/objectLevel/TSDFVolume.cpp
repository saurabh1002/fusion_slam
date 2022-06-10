#include "TSDFVolume.hpp"

#include <string>

#include "open3d/Open3D.h"

TSDFVolume::TSDFVolume(std::string class_name,
                       float class_score,
                       float voxel_size,
                       int block_resolution,
                       int est_block_count,
                       o3d::core::Tensor T_frame_to_model,
                       o3d::core::device device,
                       o3d::t::pipelines::slam::Frame sample_frame)
    : class_name_(class_name),
      class_probability_(class_score),
      model_(o3d::t::pipelines::slam::Model(voxel_size,
                                            block_resolution,
                                            est_block_count,
                                            T_frame_to_model,
                                            device)),
      input_frame_(sample_frame),
      raycast_frame_(sample_frame) {
    this->incrementExistenceCount();
}

void updateClassProbability(float detection_score) {}