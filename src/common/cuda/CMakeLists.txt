cuda_add_library(cu
    CU.cpp
)

set_property(TARGET cu
    PROPERTY POSITION_INDEPENDENT_CODE on
)
