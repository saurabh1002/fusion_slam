add_library(tsdfvolume)
target_sources(tsdfvolume PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/TSDFVolume.cpp)
target_compile_features(tsdfvolume PRIVATE cxx_std_17)
target_link_libraries(tsdfvolume PUBLIC Open3D::Open3D camera)

add_executable(map_known_poses_objects)
target_sources(map_known_poses_objects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/map_known_poses_objects.cpp)
target_link_libraries(map_known_poses_objects PUBLIC Open3D::Open3D datasets argparse::argparse yaml-cpp camera indicators::indicators)
target_include_directories(map_known_poses_objects PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes)

add_executable(map_objects_ICP_tensor)
target_sources(map_objects_ICP_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/map_frame_to_model_icp_tensor.cpp)
target_link_libraries(map_objects_ICP_tensor PUBLIC Open3D::Open3D datasets tsdfvolume argparse::argparse yaml-cpp camera indicators::indicators)
target_include_directories(map_objects_ICP_tensor PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
