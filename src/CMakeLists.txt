################################################################################
# helper libraries
################################################################################

# include directory for common header files
include_directories(common)

# add subdirectories for individual libraries
add_subdirectory(common)
add_subdirectory(external)

# presto (barycentering) and numa (hwlocutils) are GPU-free and used by the
# FITS data stack, so they build for the OpenMP backend too.
add_subdirectory(presto)
add_subdirectory(numa)


################################################################################
# plan library
################################################################################
# Single-source: FDDGPUPlan derives from GPUPlan under every backend (its cu::
# members resolve to host shims under OPENMP), so GPUPlan.cpp is always built.
add_library(plan OBJECT Plan.cpp GPUPlan.cpp)
include(CMakePrintHelpers)
cmake_print_variables(GPU_INCLUDE_DIRS)
target_include_directories(plan PUBLIC ${GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(plan PRIVATE ${GPU_DEFINITIONS})

target_link_libraries(plan PRIVATE GPU_TRACER_LIBS)

if(NOT GPU_BACKEND STREQUAL "OPENMP")

################################################################################
# dedisp c-library
################################################################################

add_subdirectory(clib)


################################################################################
# dedisp library
################################################################################
add_subdirectory(dedisp)


################################################################################
# tdd library
################################################################################
add_subdirectory(tdd)

endif() # NOT OPENMP

################################################################################
# fdd library
################################################################################
add_subdirectory(fdd)

################################################################################
# fits / data-handler library (GPU-free: FITS, filterbank, MPI datacontainer)
################################################################################
add_subdirectory(dataHandlers)
