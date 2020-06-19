cuda_add_library(dedisperse SHARED
    ${CMAKE_SOURCE_DIR}/src/dedisperse/dedisperse.cu
)

install(
    TARGETS
        dedisperse
    LIBRARY
        DESTINATION lib
)