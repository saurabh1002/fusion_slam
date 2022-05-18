add_library(datasets OBJECT)
target_sources(datasets PRIVATE freiburg1_desk.cpp)
target_compile_features(datasets PRIVATE cxx_std_17)
target_link_libraries(datasets PUBLIC Open3D::Open3D yaml-cpp)