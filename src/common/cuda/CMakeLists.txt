cuda_add_library(cu STATIC
    CU.cpp
)

# We need to link to libcuda.so to use the CUDA driver API.
# This library is part of the NVIDIA driver, but the CUDA
# toolkit contains a stub that can also be used during linking.
find_library(
    CUDA_driver_LIBRARY
    NAMES cuda
    PATHS ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES lib64/stubs
    REQUIRED
)

target_link_libraries(cu
    ${CUDA_driver_LIBRARY}
)

set_property(TARGET cu
    PROPERTY POSITION_INDEPENDENT_CODE on
)
