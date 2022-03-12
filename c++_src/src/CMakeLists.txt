add_executable(map_known_poses_full)
target_sources(map_known_poses_full PRIVATE "map_known_poses_full.cpp")
target_link_libraries(map_known_poses_full PUBLIC Open3D ${Open3D_LIBRARIES} datasets argparse::argparse yaml-cpp)
target_include_directories(map_known_poses_full PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk)