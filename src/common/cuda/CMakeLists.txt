add_library(cu SHARED
    CU.cpp
)

#find_package(CUDAnvToolsExt REQUIRED)
#find_package(CUDAToolkit REQUIRED)

find_library(ROCTRACERLIB
    NAMES roctracer64
    PATHS ${HIP_LIB_INSTALL_DIR}
    PATH_SUFFIXES lib lib64
    NO_DEFAULT_PATH
)

if(ROCTRACERLIB)
    message(STATUS "Found libroctracer64: ${ROCTRACERLIB}")
else()
    message(FATAL_ERROR "libroctracer64.so not found in ${HIP_LIB_INSTALL_DIR}")
endif()

target_link_libraries(cu
    hip::host
    hip::hipfft  # Replace roc::rocfft with hip::hipfft
    ${ROCTRACERLIB}
    # roc::roctracer # Uncomment if needed
)

