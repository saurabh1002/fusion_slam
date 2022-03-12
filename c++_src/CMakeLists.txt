# Specify minimum cmake version required
cmake_minimum_required(VERSION 3.12.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0091)
    cmake_policy(SET CMP0091 NEW)
endif()

option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
# This needs cmake_policy(SET CMP0091 NEW)
if (STATIC_WINDOWS_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# Show all warnings during compilation
set(CMAKE_CXX_FLAGS "-Wall -Wextra")

# Specify build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Make sure that it is not built in root directory, build inside /build folder
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(FATAL_ERROR "Do not build in-source. Please delete CMakeCache.txt and the CMakeFiles/ directory. Then build out-of-source.")
endif()

# CMake project description and project specific settings
project(tsdf_fusion DESCRIPTION "Fusion++ paper" LANGUAGES C CXX)

# Unit Test, Code Coverage, clang-format, clan-tidy, cppcheck Configuration:
option(ENABLE_CLANG_FORMAT "Enable static analysis with clang-format" ON)
option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
option(ENABLE_COVERAGE "Enable coverage reporting" OFF)
option(ENABLE_TEST "Enable testing" OFF)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
if (ENABLE_TEST)
  enable_testing()
endif(ENABLE_TEST)

if (ENABLE_COVERAGE)
  include(TestCoverage)
endif(ENABLE_COVERAGE)

if (ENABLE_CLANG_TIDY)
  include(StaticAnalyzers)
endif(ENABLE_CLANG_TIDY)

if (ENABLE_CLANG_FORMAT)
  include(ClangFormat)
endif(ENABLE_CLANG_FORMAT)

# Set executable and libraries output path
set(CMAKE_INCLUDE_PATH ${CMAKE_SOURCE_DIR}/include)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Use this directory to test your installation script
set(CMAKE_INSTALL_PREFIX ${PROJECT_SOURCE_DIR}/results)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
DESTINATION ${CMAKE_INSTALL_PREFIX}/include/)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/lib/
DESTINATION ${CMAKE_INSTALL_PREFIX}/lib/)

if(WIN32)
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/CMake)
else()
    find_package(Open3D HINTS ${CMAKE_INSTALL_PREFIX}/lib/cmake)
endif()
if(NOT Open3D_FOUND)
    message(FATAL_ERROR "Open3D not found, please use -DCMAKE_INSTALL_PREFIX=open3d_install_dir")
endif()

find_package(yaml-cpp REQUIRED)
find_package(argparse REQUIRED)

include_directories(${OPEN3D_INCLUDE_DIRS})

# Add subdirectory with a sub-project
add_subdirectory(${CMAKE_SOURCE_DIR}/src)
add_subdirectory(${CMAKE_SOURCE_DIR}/dataloaders)