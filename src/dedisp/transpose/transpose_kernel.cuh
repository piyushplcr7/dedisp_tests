/*
 * This file contains the implementation of transpose_kernel
 */

#ifndef TRANSPOSE_KERNEL_H_INCLUDE_GUARD
#define TRANSPOSE_KERNEL_H_INCLUDE_GUARD

//#if __CUDA_ARCH__ < 200
#define TILE_DIM            32
//#else
//#define TILE_DIM            64
//#endif
#define BLOCK_ROWS          8

typedef unsigned int gpu_size_t;

template<bool GRID_IS_POW2, typename T>
__global__
void transpose_kernel(const T* in,
                      gpu_size_t width, gpu_size_t height,
                      gpu_size_t in_stride, gpu_size_t out_stride,
                      T* out,
                      gpu_size_t block_count_x,
                      gpu_size_t block_count_y,
                      gpu_size_t log2_gridDim_y)
{
    __shared__ T tile[TILE_DIM][TILE_DIM];

    gpu_size_t blockIdx_x, blockIdx_y;

    // Do diagonal index reordering to avoid partition camping in device memory
    if( width == height ) {
        blockIdx_y = blockIdx.x;
        if( !GRID_IS_POW2 ) {
            blockIdx_x = (blockIdx.x+blockIdx.y) % gridDim.x;
        }
        else {
            blockIdx_x = (blockIdx.x+blockIdx.y) & (gridDim.x-1);
        }
    }
    else {
        gpu_size_t bid = blockIdx.x + gridDim.x*blockIdx.y;
        if( !GRID_IS_POW2 ) {
            blockIdx_y = bid % gridDim.y;
            blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
        }
        else {
            blockIdx_y = bid & (gridDim.y-1);
            blockIdx_x = ((bid >> log2_gridDim_y) + blockIdx_y) & (gridDim.x-1);
        }
    }

    // Cull excess blocks (there may be many if we round up to a power of 2)
    if( blockIdx_x >= block_count_x ||
        blockIdx_y >= block_count_y ) {
        return;
    }

    gpu_size_t index_in_x = blockIdx_x * TILE_DIM + threadIdx.x;
    gpu_size_t index_in_y = blockIdx_y * TILE_DIM + threadIdx.y;
    gpu_size_t index_in = index_in_x + (index_in_y)*in_stride;

#pragma unroll
    for( gpu_size_t i=0; i<TILE_DIM; i+=BLOCK_ROWS ) {
        // TODO: Is it possible to cull some excess threads early?
        if( index_in_x < width && index_in_y+i < height )
                tile[threadIdx.y+i][threadIdx.x] = in[index_in+i*in_stride];
    }

    __syncthreads();

    gpu_size_t index_out_x = blockIdx_y * TILE_DIM + threadIdx.x;
    // Avoid excess threads
    if( index_out_x >= height ) return;
    gpu_size_t index_out_y = blockIdx_x * TILE_DIM + threadIdx.y;
    gpu_size_t index_out = index_out_x + (index_out_y)*out_stride;

#pragma unroll
    for( gpu_size_t i=0; i<TILE_DIM; i+=BLOCK_ROWS ) {
        // Avoid excess threads
        if( index_out_y+i < width ) {
            out[index_out+i*out_stride] = tile[threadIdx.x][threadIdx.y+i];
        }
    }
}

#endif // TRANSPOSE_KERNEL_H_INCLUDE_GUARD