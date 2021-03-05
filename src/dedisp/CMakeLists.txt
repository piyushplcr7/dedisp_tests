# include directory for common header files
include_directories(${CMAKE_SOURCE_DIR}/src)

cuda_add_library(
    dedisp SHARED
    DedispPlan.cpp
    dedisperse/dedisperse.cu
    unpack/unpack.cu
    transpose/transpose.cu
)

target_link_libraries(
    dedisp
    ${CUDA_CUDART_LIBRARY}
    common
    external
    plan
)

target_include_directories(
   dedisp
   PRIVATE ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(
    dedisp
    PROPERTIES PUBLIC_HEADER DedispPlan.hpp
)

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
