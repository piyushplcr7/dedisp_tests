cuda_add_library(
    transpose SHARED
    transpose.cu
)

install(
    TARGETS
        transpose
    LIBRARY
        DESTINATION lib
)