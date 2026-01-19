################################################################################
# main
################################################################################
# cmake configuration
cmake_minimum_required(VERSION 3.21)

project(dedisp LANGUAGES C CXX)

# version number
set(DEDISP_VERSION_MAJOR 1)
set(DEDISP_VERSION_MINOR 0)
set(DEDISP_VERSION_PATCH 1)
set(DEDISP_VERSION ${DEDISP_VERSION_MAJOR}.${DEDISP_VERSION_MINOR}.${DEDISP_VERSION_PATCH} )

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

###############################################################################
# Select backend

set(GPU_BACKEND "" CACHE STRING "GPU backend (CUDA or HIP)")
set_property(CACHE GPU_BACKEND PROPERTY STRINGS CUDA HIP)

if(GPU_BACKEND STREQUAL "CUDA")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(NCCL REQUIRED)

    set(CMAKE_CUDA_ARCHITECTURES "70;80;89;90")
    set(CUDA_NVCC_FLAGS -O3 -use_fast_math -lineinfo -src-in-ptx)

    # abstracted vars
    set(GPU_INCLUDE_DIRS ${CUDAToolkit_INCLUDE_DIRS})
    set(GPU_HOST_RUNTIME_LIBS CUDA::cudart)
    set(GPU_TRACER_LIBS CUDA::nvToolsExt)
    set(GPU_LIBRARIES ${CUDAToolkit_LIBRARIES} ${NCCL_LIBRARIES})
    set(GPU_FFT_LIBS CUDA::cufft)
    set(GPU_DEFINITIONS USE_CUDA)
    set(GPU_XCCL ${NCCL_LIBRARY})
    set(GPU_XCCL_INCLUDE_DIR ${NCCL_INCLUDE_DIR})
    set(GPU_LANG CUDA)

elseif(GPU_BACKEND STREQUAL "HIP")
    enable_language(HIP)
    find_package(hip REQUIRED)
    find_package(hipfft REQUIRED)
    find_package(rocFFT REQUIRED)
    find_package(rccl REQUIRED)

    set(CMAKE_HIP_ARCHITECTURES "gfx906;gfx908;gfx90a")
    add_definitions(-D__HIP_PLATFORM_AMD__)

    # abstracted vars
    set(GPU_INCLUDE_DIRS ${HIP_INCLUDE_DIRS})
    set(GPU_HOST_RUNTIME_LIBS hip::host)
    set(GPU_LIBRARIES ${HIP_LIBRARIES})
    set(GPU_FFT_LIBS ${HIPFFT_LIBRARIES} ${ROCFFT_LIBRARIES})
    set(GPU_DEFINITIONS USE_HIP)

    find_library(ROCTRACERLIB
        NAMES roctracer64
        PATHS ${HIP_LIB_INSTALL_DIR}
        PATH_SUFFIXES lib lib64
        NO_DEFAULT_PATH
    )

    set(GPU_TRACER_LIBS ${ROCTRACERLIB})
    set(GPU_XCCL rccl)
    set(GPU_XCCL_INCLUDE_DIR ${GPU_INCLUDE_DIRS})
    set(GPU_LANG HIP)

else()
    message(FATAL_ERROR
        "GPU_BACKEND must be set to CUDA or HIP.\n"
        "Example:\n"
        "  cmake -DGPU_BACKEND=CUDA ..\n"
        "  cmake -DGPU_BACKEND=HIP  ..")
endif()

###############################################################################

# set Release as default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
  #set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "-Wall")

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Mframe")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer")

# Optimization level: -O2 (safer for profiling than -O3)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


################################################################################
# options
################################################################################
option(ENABLE_BENCHMARK "Enable benchmarking" OFF)

if (ENABLE_BENCHMARK)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEDISP_BENCHMARK")
endif()

option(ENABLE_DEBUG "Enable debug mode" OFF)

if (ENABLE_DEBUG)
    message("Enable debug option on")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DDEDISP_DEBUG")
    message("CMAKE_CXX_FLAGS = ${CMAKE_CXX_FLAGS}")
endif()

################################################################################
# OpenMP configuration
################################################################################
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

################################################################################
# CFITSIO
################################################################################
find_package(CFITSIO)

if(DEFINED CFITSIO_INCLUDE_DIR AND CFITSIO_INCLUDE_DIR)
    message(STATUS "CFITSIO include directory found: ${CFITSIO_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "CFITSIO include directory not found or invalid!")
endif()

################################################################################
# hwloc
################################################################################
find_package(HWLOC REQUIRED)

################################################################################
# MPI
################################################################################
find_package(MPI REQUIRED)

################################################################################
# FFTW
################################################################################
find_package(FFTW3 REQUIRED true)

################################################################################
# NUMA
################################################################################
find_library(NUMA_LIBRARY NAMES numa)

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
