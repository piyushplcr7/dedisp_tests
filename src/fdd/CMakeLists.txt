add_library(fdd SHARED
    FDDGPUPlan.cu
    FDDCPUPlan.cpp
    dedisperse/FDDKernel.cu
    unpack/unpack.cu
    chunk.cpp
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:plan>
    $<TARGET_OBJECTS:external>
)

##############################
# CLIG INTERFACE
##############################
add_library(fdd_gpu_clig fdd_gpu.c)

find_package(FFTW3 REQUIRED true)

target_link_libraries(fdd
    ${FFTW3F_LIBRARY}
    cu
    fdd_gpu_clig
    MPI::MPI_CXX
    ${NCCL_LIBRARY}
)

target_include_directories(fdd
   PRIVATE ${CMAKE_SOURCE_DIR}/src
   PRIVATE ${CMAKE_SOURCE_DIR}/src/fdd
   PRIVATE ${FFTW3_INCLUDE_DIR}
   PRIVATE ${NCCL_INCLUDE_DIR}
   PRIVATE ${MPI_CXX_INCLUDE_DIRS}
   PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(fdd PROPERTIES
    PUBLIC_HEADER FDDCPUPlan.hpp
    PUBLIC_HEADER FDDGPUPlan.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
    CXX_STANDARD 20
)

install(
    TARGETS
        fdd
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)
