/*
 * This file contains a CUDA implementation of the array transpose operation.
 *
 * Parts of this file are based on the transpose implementation in the
 * NVIDIA CUDA SDK.
 */

#pragma once

#include "transpose_kernel.cuh"

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
    inline U min(const U& a, const U& b) {
        return a < b ? a : b;
    }
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
    inline unsigned int log2(unsigned int a) {
        unsigned int r;
        unsigned int shift;
        r =     (a > 0xFFFF) << 4; a >>= r;
        shift = (a > 0xFF  ) << 3; a >>= shift; r |= shift;
        shift = (a > 0xF   ) << 2; a >>= shift; r |= shift;
        shift = (a > 0x3   ) << 1; a >>= shift; r |= shift;
        r |= (a >> 1);
        return r;
    }
    inline unsigned long log2(unsigned long a) {
        unsigned long r;
        unsigned long shift;
        r =     (a > 0xFFFFFFFF) << 5; a >>= r;
        shift = (a > 0xFFFF    ) << 4; a >>= shift; r |= shift;
        shift = (a > 0xFF      ) << 3; a >>= shift; r |= shift;
        shift = (a > 0xF       ) << 2; a >>= shift; r |= shift;
        shift = (a > 0x3       ) << 1; a >>= shift; r |= shift;
        r |= (a >> 1);
        return r;
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
        block_count.y = min(max_grid_dim,
                            tot_block_count.y - block_y_offset);

        for( size_t block_x_offset = 0;
             block_x_offset < tot_block_count.x;
             block_x_offset += max_grid_dim ) {

            // Handle the possibly incomplete final grid
            block_count.x = min(max_grid_dim,
                                tot_block_count.x - block_x_offset);

            // Compute the chunked parameters
            size_t x_offset = block_x_offset * TILE_DIM;
            size_t y_offset = block_y_offset * TILE_DIM;
            size_t in_offset = x_offset + y_offset*in_stride;
            size_t out_offset = y_offset + x_offset*out_stride;
            size_t w = min(max_grid_dim*TILE_DIM, width-x_offset);
            size_t h = min(max_grid_dim*TILE_DIM, height-y_offset);

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
                     log2(grid.y));
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
                     log2(grid.y));
            }

#ifndef NDEBUG
            cudaStreamSynchronize(stream);
            cudaError_t error = cudaGetLastError();
            if( error != cudaSuccess ) {
                /*
                throw std::runtime_error(
                    std::string("Transpose: CUDA error in kernel: ") +
                    cudaGetErrorString(error));
                */
            }
#endif
        }
    }
}