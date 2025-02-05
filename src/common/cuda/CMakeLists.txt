add_library(cu SHARED
    CU.cpp
)

#find_package(CUDAnvToolsExt REQUIRED)
#find_package(CUDAToolkit REQUIRED)

target_link_libraries(cu
    hip::host
    hip::hipfft  # Replace roc::rocfft with hip::hipfft
    /opt/rocm/lib/libroctracer64.so.4.1.60302
    # roc::roctracer # Uncomment if needed
)

