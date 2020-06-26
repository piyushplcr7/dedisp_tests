cuda_add_library(fdd SHARED
    FDDPlan.cpp
)

target_link_libraries(fdd
    ${CUDA_CUDART_LIBRARY}
    common
    external
    plan
)

target_include_directories(fdd
   PRIVATE ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(fdd PROPERTIES
    PUBLIC_HEADER FDDPlan.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

install(
    TARGETS
        fdd
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)