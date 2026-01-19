
#ifndef CUDAMACROHPP
#define CUDAMACROHPP

#include "gpu_runtime.hpp"

#define CUDA_CHECK(call)                                \
    do                                                  \
    {                                                   \
        const gpuError_t error_code = call;            \
        if (error_code != gpuSuccess)                  \
        {                                               \
            printf("CUDA Error:\n");                    \
            printf("    File:       %s\n", __FILE__);   \
            printf("    Line:       %d\n", __LINE__);   \
            printf("    Error code: %d\n", error_code); \
            printf("    Error text: %s\n",              \
                   gpuGetErrorString(error_code));     \
            exit(1);                                    \
        }                                               \
    } while (0)


#endif