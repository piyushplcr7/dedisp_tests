add_library(common OBJECT
    dedisp_error.cpp
    helper.cpp
)

# The cu:: wrapper library (CU.cpp) compiles under every backend now: under
# USE_OPENMP its gpu_* macros resolve to host no-ops, so FDDGPUPlan can use
# cu::DeviceMemory/Event/Marker single-source.
add_subdirectory(cuda)
