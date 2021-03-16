cuda_add_library(
    tdd SHARED
    TDDPlan.cpp
    dedisperse/TDDKernel.cu
    unpack/unpack.cu
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:plan>
    $<TARGET_OBJECTS:external>
)

target_link_libraries(
    tdd
    cu
)

target_include_directories(
    tdd
    PRIVATE ${CMAKE_SOURCE_DIR}/src
    PUBLIC ${CUDA_TOOLKIT_INCLUDE}
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
