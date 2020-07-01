/*
 * This file contains a CUDA implementation of the array transpose operation.
 *
 * Parts of this file are based on the transpose implementation in the
 * NVIDIA CUDA SDK.
 */

#pragma once

#include "transpose_kernel.cuh"

#include <algorithm>

namespace cuda_specs {
    enum { MAX_GRID_DIMENSION = 65535 };
}

template<typename T>
struct Transpose {

    Transpose() {}

    void transpose(const T* in,
                   size_t width, size_t height,
                   size_t in_stride, size_t out_stride,
                   T* out,
                   cudaStream_t stream=0);
    void transpose(const T* in,
                   size_t width, size_t height,
                   T* out,
                   cudaStream_t stream=0) {
        transpose(in, width, height, width, height, out, stream);
    }
private:
    // TODO: These should probably be imported from somewhere else
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
};

template<typename T>
void Transpose<T>::transpose(const T* in,
                             size_t width, size_t height,
                             size_t in_stride, size_t out_stride,
                             T* out,
                             cudaStream_t stream)
{
    // Parameter checks
    // TODO: Implement some sort of error returning!
    if( 0 == width || 0 == height ) return;
    if( 0 == in ) return; //throw std::runtime_error("Transpose: in is NULL");
    if( 0 == out ) return; //throw std::runtime_error("Transpose: out is NULL");
    if( width > in_stride )
        return; //throw std::runtime_error("Transpose: width exceeds in_stride");
    if( height > out_stride )
        return; //throw std::runtime_error("Transpose: height exceeds out_stride");

    // Specify thread decomposition (uses up-rounded divisions)
    dim3 tot_block_count((width-1)  / TILE_DIM + 1,
                         (height-1) / TILE_DIM + 1);

    size_t max_grid_dim = round_down_pow2((size_t)cuda_specs::MAX_GRID_DIMENSION);

    // Partition the grid into chunks that the GPU can accept at once
    for( size_t block_y_offset = 0;
         block_y_offset < tot_block_count.y;
         block_y_offset += max_grid_dim ) {

        dim3 block_count;

        // Handle the possibly incomplete final grid
        block_count.y = std::min(max_grid_dim,
                            tot_block_count.y - block_y_offset);

        for( size_t block_x_offset = 0;
             block_x_offset < tot_block_count.x;
             block_x_offset += max_grid_dim ) {

            // Handle the possibly incomplete final grid
            block_count.x = std::min(max_grid_dim,
                                tot_block_count.x - block_x_offset);

            // Compute the chunked parameters
            size_t x_offset = block_x_offset * TILE_DIM;
            size_t y_offset = block_y_offset * TILE_DIM;
            size_t in_offset = x_offset + y_offset*in_stride;
            size_t out_offset = y_offset + x_offset*out_stride;
            size_t w = std::min(max_grid_dim*TILE_DIM, width-x_offset);
            size_t h = std::min(max_grid_dim*TILE_DIM, height-y_offset);

            dim3 block(TILE_DIM, BLOCK_ROWS);

            // TODO: Unfortunately there are cases where rounding to a power of 2 becomes
            //       detrimental to performance. Could work out a heuristic.
            //bool round_grid_to_pow2 = false;
            bool round_grid_to_pow2 = true;

            // Dispatch on grid-rounding
            if (round_grid_to_pow2) {
                dim3 grid(round_up_pow2(block_count.x),
                          round_up_pow2(block_count.y));

                // Run the CUDA kernel
                transpose_kernel<true, T><<<grid, block, 0, stream>>>
                    (in + in_offset,
                     w, h,
                     in_stride, out_stride,
                     out + out_offset,
                     block_count.x,
                     block_count.y,
                     std::log2(grid.y));
            } else {
                dim3 grid(block_count.x, block_count.y);

                // Run the CUDA kernel
                transpose_kernel<false, T><<<grid, block, 0, stream>>>
                    (in + in_offset,
                     w, h,
                     in_stride, out_stride,
                     out + out_offset,
                     block_count.x,
                     block_count.y,
                     std::log2(grid.y));
            }
        }
    }
}