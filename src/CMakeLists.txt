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

target_link_libraries(plan PUBLIC
    ${CUDA_CUDART_LIBRARY}
    common
    external
)

target_include_directories(plan
   PUBLIC ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(plan PROPERTIES
    PUBLIC_HEADER Plan.hpp
    POSITION_INDEPENDENT_CODE on
)

install(
    TARGETS
        plan
    PUBLIC_HEADER
        DESTINATION include
)


################################################################################
# tdd library
################################################################################

add_subdirectory(tdd)


################################################################################
# fdd library
################################################################################

add_subdirectory(fdd)