################################################################################
# main
################################################################################

project(dedisp)

# version number
set(DEDISP_VERSION_MAJOR 1)
set(DEDISP_VERSION_MINOR 0)
set(DEDISP_VERSION_PATCH 1)
set(DEDISP_VERSION ${DEDISP_VERSION_MAJOR}.${DEDISP_VERSION_MINOR}.${DEDISP_VERSION_PATCH} )

cmake_minimum_required(VERSION 3.16)

# dependencies
find_package(CUDA)

# set Release as default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# set default compiler flags
set(CMAKE_CXX_FLAGS "-Wall -g -O3")


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
# dedisp library
################################################################################

cuda_add_library(dedisp SHARED src/dedisp.cu)

target_include_directories(dedisp PUBLIC src)

target_link_libraries(dedisp ${CUDA_CUDART_LIBRARY})

set_target_properties(dedisp PROPERTIES PUBLIC_HEADER "src/dedisp.h;src/DedispPlan.hpp")

set_target_properties(
    dedisp PROPERTIES
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

install(
    TARGETS
        dedisp
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)

# make sure that all subsequent binaries can find the dedisp library
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


################################################################################
# dedisp_fil binary
################################################################################

add_executable(dedisp_fil bin/dedisp_fil/dedisp_fil.c)

target_include_directories(dedisp_fil PRIVATE src)

target_link_libraries(dedisp_fil dedisp m)

install(
    TARGETS
        dedisp_fil
    RUNTIME
        DESTINATION bin
)


################################################################################
# python binaries
################################################################################

install(
    FILES
        python/filterbank.py
        python/plot_burst.py
        python/psr_constants.py
        python/sigproc.py
    DESTINATION bin
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_READ  GROUP_EXECUTE
        WORLD_READ WORLD_READ  WORLD_EXECUTE
)