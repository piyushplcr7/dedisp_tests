cuda_add_library(cu
    CU.cpp
)

find_package(CUDAnvToolsExt REQUIRED)

target_link_libraries(cu
    ${CUDA_nvToolsExt_LIBRARY}
)

set_property(TARGET cu
    PROPERTY POSITION_INDEPENDENT_CODE on
)
