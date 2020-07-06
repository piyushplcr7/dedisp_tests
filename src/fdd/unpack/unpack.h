#include "dedisp_types.h"

#include "cuda_runtime.h"

void transpose_unpack(
    const dedisp_word* d_in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    dedisp_word* d_out,
    dedisp_size in_nbits, dedisp_size out_nbits,
    float scale,
    cudaStream_t stream);
