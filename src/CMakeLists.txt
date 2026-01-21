################################################################################
# helper libraries
################################################################################

# include directory for common header files
include_directories(common)

# add subdirectories for individual libraries
add_subdirectory(common)
add_subdirectory(external)

add_subdirectory(numa)


################################################################################
# plan library
################################################################################
add_library(plan OBJECT Plan.cpp GPUPlan.cpp)
include(CMakePrintHelpers)
cmake_print_variables(GPU_INCLUDE_DIRS)
target_include_directories(plan PUBLIC ${GPU_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(plan PRIVATE ${GPU_DEFINITIONS})

target_link_libraries(plan PRIVATE GPU_TRACER_LIBS)

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


################################################################################
# fdd library
################################################################################
add_subdirectory(fdd)


################################################################################
# fts library
################################################################################
add_subdirectory(fits)