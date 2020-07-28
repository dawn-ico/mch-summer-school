cmake_minimum_required(VERSION 3.13)

project(icondusk-e2e LANGUAGES CXX C)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS "CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")

#include(CompileAsCUDA)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 14)

set(CUDA_ARCH "sm_70" CACHE STRING "CUDA architecture (e.g. sm_35, sm_60, sm_70)")

if(DEFINED dawn4py_DIR)
  find_path(dawn4py_spd  dawn4py PATHS ${dawn4py_DIR} PATH_SUFFIXES lib/python3.8/site-packages/)

  if(NOT dawn4py_spd)
    message(FATAL_ERROR "dawn4py_DIR directory does not contain a dawn4py installation for python 3.8")
  endif()

  set(dawn4py_rootdir ${dawn4py_spd}/dawn4py)

  find_path(dawn4py_INCLUDE_DIR driver-includes PATH ${dawn4py_rootdir}/_external_src)
endif()

option(ENABLE_PERFORMANCE_METERS "compile with meters enabled" ON)
set(PRECISION "double" CACHE STRING "")

### Require out-of-source builds
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please make a build subdirectory. Feel free to remove CMakeCache.txt and CMakeFiles.")
endif()

include(CTest)

find_package(Python3 COMPONENTS Interpreter)
find_package(atlas CONFIG REQUIRED) 
find_package(atlas_utils CONFIG REQUIRED)
include(FetchGoogleTest)
include(FetchDusk)

enable_testing()

configure_file(${CMAKE_SOURCE_DIR}/scripts/calldusk.sh.in ${CMAKE_SOURCE_DIR}/scripts/tmp/calldusk.sh @ONLY)
file(
    COPY  ${CMAKE_SOURCE_DIR}/scripts/tmp/calldusk.sh
    DESTINATION ${CMAKE_SOURCE_DIR}/scripts/
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    )    

set(dusk_tests
  nh_diffusion_fvm
)

add_subdirectory(dusk_stencils)
add_subdirectory(cpp_drivers)