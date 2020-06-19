cuda_add_library(transpose SHARED
    ${CMAKE_SOURCE_DIR}/src/transpose/transpose.cu
)

install(
    TARGETS
        transpose
    LIBRARY
        DESTINATION lib
)