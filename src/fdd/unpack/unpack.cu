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
    const dedisp_word* d_in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    dedisp_word* d_out,
    dedisp_size in_nbits, dedisp_size out_nbits,
    float scale,
    cudaStream_t stream)
{
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

            // Run the CUDA kernel
            transpose_unpack_kernel<dedisp_word><<<grid, block, 0, stream>>> \
                (d_in + in_offset,      \
                 w, h,                  \
                 in_stride, out_stride, \
                 d_out + out_offset,    \
                 block_count.x,         \
                 block_count.y,         \
                 in_nbits,              \
                 scale);
        } // end for block_x_offset
    } // end for block_y_offset
}