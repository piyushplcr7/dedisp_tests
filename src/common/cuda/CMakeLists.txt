add_library(cu
    CU.cpp
)

target_link_libraries(cu
    ${GPU_HOST_RUNTIME_LIBS}
    ${GPU_TRACER_LIBS}
    ${GPU_FFT_LIBS}
)
