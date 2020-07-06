cuda_add_library(fdd SHARED
    FDDPlan.cpp
    dedisperse/FDDKernel.cu
    unpack/unpack.cu
)

find_package(FFTW3 REQUIRED true)


target_link_libraries(fdd
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_cufft_LIBRARY}
    ${FFTW3F_LIBRARY}
    common
    external
    plan
)

target_include_directories(fdd
   PRIVATE ${CMAKE_SOURCE_DIR}/src
   PRIVATE ${CMAKE_SOURCE_DIR}/src/dedisp
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