cuda_add_library(dedisp SHARED
    TDDPlan.cpp
    dedisperse/DedispKernel.cu
    unpack/unpack.cu
)

target_link_libraries(dedisp
    ${CUDA_CUDART_LIBRARY}
    common
    external
    plan
)

target_include_directories(dedisp
   PRIVATE ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(dedisp
    PROPERTIES PUBLIC_HEADER TDDPlan.hpp
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