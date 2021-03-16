cuda_add_library(cu
    CU.cpp
)

find_package(CUDAnvToolsExt REQUIRED)

target_link_libraries(cu
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_nvToolsExt_LIBRARY}
    ${CUDA_cufft_LIBRARY}
)
