################################################################################
# helper libraries
################################################################################

# include directory for common header files

include_directories(common)

# add subdirectories for individual libraries
add_subdirectory(common)
add_subdirectory(transpose)
add_subdirectory(dedisperse)
add_subdirectory(unpack)
add_subdirectory(external)


################################################################################
# dedisp library
################################################################################

add_library(dedisp SHARED
    dedisp_kernels.cpp
    DedispPlan.cpp
)

target_link_libraries(dedisp
    ${CUDA_CUDART_LIBRARY}
    common
    external
    transpose
    dedisperse
    unpack)

set_target_properties(dedisp
    PROPERTIES PUBLIC_HEADER DedispPlan.hpp)

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
