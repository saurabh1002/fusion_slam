add_executable(map_known_poses_objects)
target_sources(map_known_poses_objects PUBLIC "map_known_poses_objects.cpp")
target_link_libraries(map_known_poses_objects PUBLIC ${Open3D_LIBRARIES} datasets argparse::argparse yaml-cpp camera)
target_include_directories(map_known_poses_objects PUBLIC ${CMAKE_SOURCE_DIR}/dataloaders/freiburg1_desk ${CMAKE_SOURCE_DIR}/includes)
