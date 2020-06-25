cuda_add_library(dedisperse SHARED
    ${CMAKE_SOURCE_DIR}/src/dedisperse/dedisperse.cu
)

target_include_directories(dedisperse
    PRIVATE ${CMAKE_SOURCE_DIR}/src/common/cuda
)

install(
    TARGETS
        dedisperse
    LIBRARY
        DESTINATION lib
)