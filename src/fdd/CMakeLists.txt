set_source_files_properties(
    dedisperse/FDDKernel.cu
    unpack/unpack.cu
    PROPERTIES LANGUAGE ${GPU_LANG}
)

add_library(fdd SHARED
    FDDGPUPlan.cpp
    FDDCPUPlan.cpp
    dedisperse/FDDKernel.cu
    unpack/unpack.cu
    chunk.cpp
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:plan>
    $<TARGET_OBJECTS:external>
)

target_include_directories(fdd PUBLIC 
    ${CMAKE_SOURCE_DIR}/src/gpu_macros 
    ${CMAKE_SOURCE_DIR}/src/numa
    ${CMAKE_SOURCE_DIR}/src/fits
    ${CFITSIO_INCLUDE_DIR}
    )

target_compile_definitions(fdd PRIVATE ${GPU_DEFINITIONS})
target_link_libraries(fdd PRIVATE GPU_TRACER_LIBS hwlocutils fits ${CFITSIO_LIBRARY} datacontainer)

##############################
# CLIG INTERFACE
##############################
add_library(fdd_gpu_clig fdd_gpu.c)

target_link_libraries(fdd PRIVATE
    ${FFTW3F_LIBRARY}
    cu
    fdd_gpu_clig
    MPI::MPI_CXX
    GPU_XCCL
    ${NUMA_LIBRARY}
)

target_include_directories(fdd
   PRIVATE ${CMAKE_SOURCE_DIR}/src
   PRIVATE ${CMAKE_SOURCE_DIR}/src/fdd
   PRIVATE ${FFTW3_INCLUDE_DIR}
   PRIVATE ${NCCL_INCLUDE_DIR}
   PRIVATE ${MPI_CXX_INCLUDE_DIRS}
   PUBLIC ${GPU_INCLUDE_DIRS}
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
