#include "gpu_runtime.hpp"
// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <algorithm>

#include "dedisp_types.h"

#include "unpack_kernel.cuh"

template<typename U>
inline U round_up_pow2(const U& a) {
    U r = a-1;
    for( unsigned long i=1; i<=sizeof(U)*8/2; i<<=1 ) r |= r >> i;
    return r+1;
}

template<typename U>
inline U round_down_pow2(const U& a) {
    return round_up_pow2(a+1)/2;
}

void transpose_unpack(
    const float* d_in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    float* d_out,
    dedisp_size in_nbits, dedisp_size out_nbits,
    float scale,
    gpuStream_t stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    // Specify thread decomposition (uses up-rounded divisions)
    dim3 tot_block_count((width-1)  / TILE_DIM + 1,
                         (height-1) / TILE_DIM + 1);

    size_t max_grid_dim = round_down_pow2(32768);

     // Partition the grid into chunks that the GPU can accept at once
    for (size_t block_y_offset = 0;
                block_y_offset < tot_block_count.y;
                block_y_offset += max_grid_dim)
    {

        dim3 block_count;

        // Handle the possibly incomplete final grid
        block_count.y = std::min(max_grid_dim, tot_block_count.y - block_y_offset);

        for (size_t block_x_offset = 0;
                    block_x_offset < tot_block_count.x;
                    block_x_offset += max_grid_dim)
        {
            // Handle the possibly incomplete final grid
            block_count.x = std::min(max_grid_dim, tot_block_count.x - block_x_offset);

            // Compute the chunked parameters
            size_t x_offset = block_x_offset * TILE_DIM;
            size_t y_offset = block_y_offset * TILE_DIM;
            size_t in_offset = x_offset + y_offset*in_stride;
            size_t out_offset = y_offset + x_offset*out_stride;
            size_t w = std::min(max_grid_dim*TILE_DIM, width-x_offset);
            size_t h = std::min(max_grid_dim*TILE_DIM, height-y_offset);

            dim3 block(TILE_DIM, BLOCK_ROWS);

            // Specify grid decomposition
            dim3 grid(round_up_pow2(block_count.x),
                      round_up_pow2(block_count.y));

            if (in_nbits == 8) {
                // Run the transpose unpack kernel for 8-bit input
                // Run the CUDA kernel
                transpose_unpack_kernel<dedisp_word><<<grid, block, 0, stream>>> \
                    ((const dedisp_word*)d_in + in_offset,      \
                    w, h,                  \
                    in_stride, out_stride, \
                    (dedisp_word*)d_out + out_offset,    \
                    block_count.x,         \
                    block_count.y,         \
                    in_nbits,              \
                    scale);
            }
            else if (in_nbits == 32) {
                // Run the transpose kernel for 32-bit input (no unpacking)
                // Run the CUDA kernel
                transpose_kernel<float><<<grid, block, 0, stream>>> \
                    ((const float*)d_in + in_offset,      \
                    w, h,                  \
                    in_stride, out_stride, \
                    (float*)d_out + out_offset,    \
                    block_count.x,         \
                    block_count.y,         \
                    in_nbits,              \
                    scale);
            }

        } // end for block_x_offset
    } // end for block_y_offset
#elif defined(USE_OPENMP)
    if (in_nbits == 32) {
        // Pure transpose matching transpose_kernel<float>:
        //   input layout:  in[height][width]  i.e. in[time][chan], row-stride = in_stride
        //   output layout: out[width][height] i.e. out[chan][time], row-stride = out_stride
        //   => d_out[x * out_stride + y] = d_in[y * in_stride + x]
        //      where x = chan index, y = time index
        #pragma omp parallel for
        for (size_t x = 0; x < width; ++x)
            for (size_t y = 0; y < height; ++y)
                d_out[x * out_stride + y] = d_in[y * in_stride + x];
    } else if (in_nbits == 8) {
        // 8-bit packed words -> EXPANSION channels each, (val-127.5)*scale.
        // input:  in[time][chan_word], row-stride = in_stride
        // output: out[chan][time],     row-stride = out_stride  (chan = x*EXPANSION+j)
        const dedisp_word* in_w = reinterpret_cast<const dedisp_word*>(d_in);
        dedisp_size in_mask = (1u << in_nbits) - 1;
        #pragma omp parallel for
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                dedisp_word word = in_w[y * in_stride + x];
                for (unsigned j = 0; j < EXPANSION; ++j) {
                    dedisp_word val = (word >> (j * in_nbits)) & in_mask;
                    float result = (((float)val) - 127.5f) * scale;
                    d_out[(x * EXPANSION + j) * out_stride + y] = result;
                }
            }
        }
    }
#endif
}