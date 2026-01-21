add_library(cu
    CU.cpp
)

target_link_libraries(cu PRIVATE
    ${GPU_HOST_RUNTIME_LIBS}
    ${GPU_FFT_LIBS}
)

target_link_libraries(cu PRIVATE GPU_TRACER_LIBS)

target_include_directories(cu PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(cu PRIVATE ${GPU_DEFINITIONS})
