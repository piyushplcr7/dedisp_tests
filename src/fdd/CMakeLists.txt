add_library(fdd SHARED
    FDDGPUPlan.cpp
    FDDCPUPlan.cpp
    dedisperse/FDDKernel.cu
    unpack/unpack.cu
    chunk.cpp
    $<TARGET_OBJECTS:common>
    $<TARGET_OBJECTS:plan>
    $<TARGET_OBJECTS:external>
)

find_package(FFTW3 REQUIRED true)

target_link_libraries(fdd
    ${FFTW3F_LIBRARY}
    cu
)

target_include_directories(fdd
   PRIVATE ${CMAKE_SOURCE_DIR}/src
   PRIVATE ${CMAKE_SOURCE_DIR}/src/fdd
   PRIVATE ${FFTW3_INCLUDE_DIR}
   PUBLIC ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(fdd PROPERTIES
    PUBLIC_HEADER FDDCPUPlan.hpp
    PUBLIC_HEADER FDDGPUPlan.hpp
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
