cuda_add_library(fdd SHARED
    FDDPlan.cpp
)

find_package(FFTW3 REQUIRED true)

find_package(OpenMP)

if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_link_libraries(fdd
    ${CUDA_CUDART_LIBRARY}
    ${FFTW3F_LIBRARY}
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