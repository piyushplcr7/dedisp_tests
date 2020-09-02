cuda_add_library(
    tdd SHARED
    TDDPlan.cpp
    dedisperse/TDDKernel.cu
    unpack/unpack.cu
)

target_link_libraries(
    tdd
    ${CUDA_CUDART_LIBRARY}
    common
    external
    plan
)

target_include_directories(
    tdd
   PRIVATE ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(
    tdd
    PROPERTIES PUBLIC_HEADER TDDPlan.hpp
)

set_target_properties(
    tdd PROPERTIES
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

install(
    TARGETS
        tdd
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)