cuda_add_library(
    dedisperse SHARED
    dedisperse.cu
)

install(
    TARGETS
        dedisperse
    LIBRARY
        DESTINATION lib
)