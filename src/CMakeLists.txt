################################################################################
# helper libraries
################################################################################

# include directory for common header files
include_directories(common)

# add subdirectories for individual libraries
add_subdirectory(common)
add_subdirectory(external)


################################################################################
# plan library
################################################################################
add_library(plan OBJECT Plan.cpp GPUPlan.cpp)
include(CMakePrintHelpers)
cmake_print_variables(CUDAToolkit_INCLUDE_DIRS)
target_include_directories(plan PUBLIC ${CUDAToolkit_INCLUDE_DIRS})


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
