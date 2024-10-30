add_library(cu
    CU.cpp
)

#find_package(CUDAnvToolsExt REQUIRED)
#find_package(CUDAToolkit REQUIRED)

target_link_libraries(cu
    CUDA::cudart
    CUDA::nvToolsExt
    CUDA::cufft
)
