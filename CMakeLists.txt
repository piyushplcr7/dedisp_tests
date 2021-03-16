################################################################################
# main
################################################################################

project(dedisp)

# version number
set(DEDISP_VERSION_MAJOR 1)
set(DEDISP_VERSION_MINOR 0)
set(DEDISP_VERSION_PATCH 1)
set(DEDISP_VERSION ${DEDISP_VERSION_MAJOR}.${DEDISP_VERSION_MINOR}.${DEDISP_VERSION_PATCH} )

# cmake configuration
cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# dependencies
find_package(CUDA)

# set Release as default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set default compiler flags
set(CMAKE_CXX_FLAGS "-Wall -g -O3")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


################################################################################
# options
################################################################################
option(ENABLE_BENCHMARK on)

if (ENABLE_BENCHMARK)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEDISP_BENCHMARK")
endif()

option(ENABLE_DEBUG off)

if (ENABLE_DEBUG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEDISP_DEBUG")
endif()


################################################################################
# cuda configuration
################################################################################

set(CUDA_NVCC_FLAGS -O3 -use_fast_math -lineinfo -src-in-ptx)

# compile for specified architecture (if any)
set(CUDA_ARCH "" CACHE STRING "CUDA architecture")

if (${CUDA_ARCH})
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -arch=compute_${CUDA_ARCH} -code=compute_${CUDA_ARCH})
endif()


################################################################################
# OpenMP configuration
################################################################################
find_package(OpenMP)

if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


################################################################################
# library
################################################################################
add_subdirectory(src)


################################################################################
# binaries
################################################################################
add_subdirectory(bin)


################################################################################
# python binaries
################################################################################
add_subdirectory(python)


################################################################################
# tests
################################################################################
enable_testing()
add_test(NAME tDedisp COMMAND testdedisp)
