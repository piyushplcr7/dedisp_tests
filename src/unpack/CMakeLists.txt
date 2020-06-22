cuda_add_library(unpack SHARED
    ${CMAKE_SOURCE_DIR}/src/unpack/unpack.cu
)

install(
    TARGETS
        unpack
    LIBRARY
        DESTINATION lib
)