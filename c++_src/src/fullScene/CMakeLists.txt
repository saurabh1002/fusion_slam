add_executable(map_known_poses_full)
target_sources(map_known_poses_full PUBLIC "map_known_poses_full.cpp")
target_link_libraries(map_known_poses_full PUBLIC Open3D::Open3D datasets argparse::argparse yaml-cpp camera indicators::indicators)
target_include_directories(map_known_poses_full PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes)

add_executable(model_to_frame_icp_full)
target_sources(model_to_frame_icp_full PUBLIC "map_frame_to_model_icp.cpp")
target_link_libraries(model_to_frame_icp_full PUBLIC Open3D::Open3D datasets argparse::argparse yaml-cpp camera indicators::indicators)
target_include_directories(model_to_frame_icp_full PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes)

add_executable(model_to_frame_icp_full_tensor)
target_sources(model_to_frame_icp_full_tensor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/map_frame_to_model_icp_tensor.cpp)
target_link_libraries(model_to_frame_icp_full_tensor PUBLIC Open3D::Open3D datasets argparse::argparse yaml-cpp camera indicators::indicators)
target_include_directories(model_to_frame_icp_full_tensor PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes)