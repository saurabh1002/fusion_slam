add_library(camera)
target_sources(camera PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/camera.cpp)
target_link_libraries(camera PUBLIC ${Open3D_LIBRARIES} yaml-cpp)
target_include_directories(camera PUBLIC ${CMAKE_SOURCE_DIR}/includes)