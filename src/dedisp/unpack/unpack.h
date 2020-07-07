#include "dedisp_types.h"

#include "cuda_runtime.h"

void unpack(const dedisp_word* d_transposed,
            dedisp_size nsamps, dedisp_size nchan_words,
            dedisp_word* d_unpacked,
            dedisp_size in_nbits, dedisp_size out_nbits,
            cudaStream_t stream = 0);

void transpose_unpack(
    const dedisp_word* d_in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    dedisp_word* d_out,
    dedisp_size in_nbits, dedisp_size out_nbits,
    cudaStream_t stream);
