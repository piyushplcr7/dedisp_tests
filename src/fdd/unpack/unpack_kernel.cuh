#define TILE_DIM     32
#define BLOCK_ROWS   8
#define EXPANSION    4

template<typename WordType>
__global__
void transpose_unpack_kernel(
    const WordType* in,
    size_t width,
    size_t height,
    size_t in_stride,
    size_t out_stride,
    WordType* out,
    size_t block_count_x,
    size_t block_count_y,
    size_t in_nbits,
    float scale)
{
    // Cull excess blocks
    if (blockIdx.x >= block_count_x ||
        blockIdx.y >= block_count_y)
    {
        return;
    }

    // Unpack 32-bit words into expansion in-nbits-bit values and convert these to float
    // WordType in[width][height] -> float out[height*EXPANSION][width]
    #pragma unroll
    for (size_t i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        // Compute index in input matrix data[height][width]
        size_t index_in_x = blockIdx.x * TILE_DIM + threadIdx.x;
        size_t index_in_y = blockIdx.y * TILE_DIM + threadIdx.y + i;
        size_t index_in   = index_in_x + index_in_y*in_stride;

        if (index_in_x < width && index_in_y < height)
        {
            for (unsigned int j = 0; j < EXPANSION; j++)
            {
                // Compute index in output matrix data[width*4][height]
                size_t index_out_x = blockIdx.y * TILE_DIM + threadIdx.y + i;
                size_t index_out_y = blockIdx.x * (TILE_DIM*EXPANSION) +
                                               (threadIdx.x*EXPANSION) + j;
                size_t index_out   = index_out_x + index_out_y*out_stride;

                // Load input word
                WordType word = in[index_in];

                // Extract value from word
                WordType in_mask = (1<<in_nbits)-1;
                int k_in = j * in_nbits;
                WordType val = (word >> k_in) & in_mask;

                // Convert to float and write the result to device memory
                float *dst_ptr = (float *) &out[index_out];
                *dst_ptr = (((float) val) - 127.5f) * scale;
            }
        }
    }
}