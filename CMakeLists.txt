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
# stopwatch class
################################################################################
add_library(stopwatch OBJECT src/external/Stopwatch.cpp)
set_property(TARGET stopwatch PROPERTY POSITION_INDEPENDENT_CODE on)


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
# transpose library
################################################################################

cuda_add_library(transpose SHARED src/transpose/transpose.cu)

install(
    TARGETS
        transpose
    LIBRARY
        DESTINATION lib
)


################################################################################
# dedisperse library
################################################################################

cuda_add_library(dedisperse SHARED src/dedisperse/dedisperse.cu)

target_include_directories(dedisperse PUBLIC
    src
)

install(
    TARGETS
        dedisperse
    LIBRARY
        DESTINATION lib
)


################################################################################
# dedisp library
################################################################################

cuda_add_library(dedisp SHARED
    src/dedisp_error.cpp
    src/dedisp_kernels.cpp
    src/DedispPlan.cu
)

target_include_directories(dedisp PUBLIC
    src
)

target_link_libraries(dedisp
    ${CUDA_CUDART_LIBRARY}
    stopwatch
    transpose
    dedisperse)

set_target_properties(dedisp PROPERTIES PUBLIC_HEADER src/DedispPlan.hpp)

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

add_executable(dedisp_fil bin/dedisp_fil/dedisp_fil.cpp)

target_include_directories(dedisp_fil PRIVATE src)

target_link_libraries(dedisp_fil dedisp m)

install(
    TARGETS
        dedisp_fil
    RUNTIME
        DESTINATION bin
)


################################################################################
# testdedisp binary
################################################################################

add_executable(testdedisp bin/testdedisp/testdedisp.cpp)

target_include_directories(testdedisp PRIVATE src)

target_link_libraries(testdedisp dedisp m)

install(
    TARGETS
        testdedisp
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


################################################################################
# tests
################################################################################
enable_testing()
add_test(NAME tDedisp COMMAND testdedisp)
