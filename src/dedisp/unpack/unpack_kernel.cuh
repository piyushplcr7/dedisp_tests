#define TILE_DIM     32
#define BLOCK_ROWS   8

template<typename WordType, unsigned int OUT_NBITS>
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
    size_t in_nbits)
{
    // The input data has dimensions height * width,
    // with width corresponding to a number of channel words,
    // and height corresponding to a number of observing channels.
    // The output will have expansion times as many channel words,
    // thus this kernel transforms:
    // data[height][width] -> unpacked[width][height*expansion]
    dedisp_size expansion = OUT_NBITS / in_nbits;
    // the output stride therefore scales linearly with the expansion.
    out_stride *= expansion;

    // Shared memory buffer for the transposed input data
    __shared__ WordType tile[TILE_DIM][TILE_DIM];

    // Cull excess blocks
    if (blockIdx.x >= block_count_x ||
        blockIdx.y >= block_count_y)
    {
        return;
    }

    // Compute index in input matrix data[height][width]
    size_t index_in_x = blockIdx.x * TILE_DIM + threadIdx.x;
    size_t index_in_y = blockIdx.y * TILE_DIM + threadIdx.y;
    size_t index_in   = index_in_x + (index_in_y)*in_stride;

    // Transpose a tile of input into shared memory:
    // data[nwords_tile][nsamps_tile] -> transposed[nsamps_tile][nwords_tile]
    #pragma unroll
    for (size_t i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (index_in_x < width && index_in_y+i < height)
        {
            tile[threadIdx.x][threadIdx.y+i] = in[index_in+i*in_stride];
        }
    }

    __syncthreads();

    // Helper variables for data unpacking
    int out_chans_per_word = sizeof(WordType)*8 / OUT_NBITS;
    int in_chans_per_word = sizeof(WordType)*8 / in_nbits;
    int norm = ((1l<<OUT_NBITS)-1) / ((1l<<in_nbits)-1);
    WordType in_mask  = (1<<in_nbits)-1;
    WordType out_mask = (1<<OUT_NBITS)-1;

    // Unpack the inner dimension of the transposed tile:
    // transposed[nsamps_tile][nwords_tile] -> unpacked[nsamps_tile][nwords_tile*expansion]
    // Process one row (one samp) per iteration.
    #pragma unroll
    for (size_t i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        // Map threads [0:tile_dim] to [0:tile_dim*expansion],
        // to compute [nwords_tile*expansion] channel words.
        unsigned int tile_width_out = TILE_DIM * expansion;
        for (size_t j = 0; j < tile_width_out; j += blockDim.x)
        {
            // Compute index in output matrix unpacked[width][height*expansion]
            size_t index_out_x = blockIdx.y * tile_width_out + (threadIdx.x + j);
            size_t index_out_y = blockIdx.x * TILE_DIM + threadIdx.y;
            size_t index_out   = index_out_x + (index_out_y)*out_stride;

            // Avoid excess threads
            if (index_out_x < out_stride && index_out_y+i < width)
            {
                // Construct an output channel word
                WordType result = 0;
                for (int k = 0; k < sizeof(WordType)*8; k += OUT_NBITS)
                {
                    int out_cw = threadIdx.x + j;
                    int c = out_cw * out_chans_per_word + k/OUT_NBITS;
                    int in_cw = c / in_chans_per_word;
                    int in_k  = c % in_chans_per_word * in_nbits;
                    WordType word = tile[threadIdx.y+i][in_cw];

                    WordType val = (word >> in_k) & in_mask;
                    result |= ((val * norm) & out_mask) << k;
                }

                // Write the result to device memory
                out[index_out+i*out_stride] = result;
            }
        }
    }
}