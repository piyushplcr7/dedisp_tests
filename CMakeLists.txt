################################################################################
# main
################################################################################

project(dedisp)

# version number
set(DEDISP_VERSION_MAJOR 1)
set(DEDISP_VERSION_MINOR 0)
set(DEDISP_VERSION_PATCH 1)
set(DEDISP_VERSION ${DEDISP_VERSION_MAJOR}.${DEDISP_VERSION_MINOR}.${DEDISP_VERSION_PATCH} )

cmake_minimum_required(VERSION 2.8)

# dependencies
find_package(CUDA)

# set Release as default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set default compiler flags
set(CMAKE_CXX_FLAGS "-Wall -g -O3")


################################################################################
# options
################################################################################
option(ENABLE_BENCHMARK on)

if (ENABLE_BENCHMARK)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEDISP_BENCHMARK")
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
