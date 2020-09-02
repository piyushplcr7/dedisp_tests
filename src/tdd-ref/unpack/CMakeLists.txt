cuda_add_library(
    unpack SHARED
    unpack.cu
)

install(
    TARGETS
        unpack
    LIBRARY
        DESTINATION lib
)