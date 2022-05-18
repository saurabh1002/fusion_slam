add_executable(map_known_poses_full)
target_sources(map_known_poses_full PUBLIC "map_known_poses_full.cpp")
target_link_libraries(map_known_poses_full PUBLIC ${Open3D_LIBRARIES} datasets argparse::argparse yaml-cpp camera)
target_include_directories(map_known_poses_full PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes)

add_executable(model_to_frame_icp_full)
target_sources(model_to_frame_icp_full PUBLIC "map_frame_to_model_icp.cpp")
target_link_libraries(model_to_frame_icp_full PUBLIC ${Open3D_LIBRARIES} datasets argparse::argparse yaml-cpp camera)
target_include_directories(model_to_frame_icp_full PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes)