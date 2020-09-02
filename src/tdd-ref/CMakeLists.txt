# include directory for common header files
include_directories(${CMAKE_SOURCE_DIR}/src)

# add subdirectories for individual libraries
add_subdirectory(transpose)
add_subdirectory(dedisperse)
add_subdirectory(unpack)

add_library(dedisp SHARED
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
