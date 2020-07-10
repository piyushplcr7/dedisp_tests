#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "dedisp_types.h"

template<typename WordType>
struct unpack_functor
    : public thrust::unary_function<unsigned int,WordType> {
    const WordType* in;
    int             nsamps;
    int             in_nbits;
    int             out_nbits;
    unpack_functor(const WordType* in_, int nsamps_, int in_nbits_, int out_nbits_)
        : in(in_), nsamps(nsamps_), in_nbits(in_nbits_), out_nbits(out_nbits_) {}
    inline __host__ __device__
    WordType operator()(unsigned int i) const {
        int out_chans_per_word = sizeof(WordType)*8 / out_nbits;
        int in_chans_per_word = sizeof(WordType)*8 / in_nbits;
        //int expansion = out_nbits / in_nbits;
        int norm = ((1l<<out_nbits)-1) / ((1l<<in_nbits)-1);
        WordType in_mask  = (1<<in_nbits)-1;
        WordType out_mask = (1<<out_nbits)-1;

        /*
          cw\k 0123 0123
          0    0123|0123
          1    4567|4567

          cw\k 0 1
          0    0 1 | 0 1
          1    2 3 | 2 3
          2    4 5 | 4 5
          3    6 7 | 6 7


         */

        unsigned int t      = i % nsamps;
        // Find the channel word indices
        unsigned int out_cw = i / nsamps;
        //unsigned int in_cw  = out_cw / expansion;
        //unsigned int in_i   = in_cw * nsamps + t;
        //WordType word = in[in_i];

        WordType result = 0;
        for( int k=0; k<sizeof(WordType)*8; k+=out_nbits ) {

            int c = out_cw * out_chans_per_word + k/out_nbits;
            int in_cw = c / in_chans_per_word;
            int in_k  = c % in_chans_per_word * in_nbits;
            int in_i  = in_cw * nsamps + t;
            WordType word = in[in_i];

            WordType val = (word >> in_k) & in_mask;
            result |= ((val * norm) & out_mask) << k;
        }
        return result;
    }
};
void unpack(const dedisp_word* d_transposed,
            dedisp_size nsamps, dedisp_size nchan_words,
            dedisp_word* d_unpacked,
            dedisp_size in_nbits, dedisp_size out_nbits,
            cudaStream_t stream)
{
    dedisp_size expansion = out_nbits / in_nbits;
    dedisp_size in_count  = nsamps * nchan_words;
    dedisp_size out_count = in_count * expansion;

    using thrust::make_counting_iterator;

    thrust::transform(thrust::cuda::par.on(stream),
                      make_counting_iterator<unsigned int>(0),
                      make_counting_iterator<unsigned int>(out_count),
                      d_unpacked,
                      unpack_functor<dedisp_word>(d_transposed, nsamps,
                                                  in_nbits, out_nbits));
}

#define TILE_DIM     32
#define BLOCK_ROWS   8

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
    size_t out_nbits)
{
    // The input data has dimensions height * width,
    // with width corresponding to a number of channel words,
    // and height corresponding to a number of observing channels.
    // The output will have expansion times as many channel words,
    // thus this kernel transforms:
    // data[height][width] -> unpacked[width][height*expansion]
    dedisp_size expansion = out_nbits / in_nbits;
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
    int out_chans_per_word = sizeof(WordType)*8 / out_nbits;
    int in_chans_per_word = sizeof(WordType)*8 / in_nbits;
    int norm = ((1l<<out_nbits)-1) / ((1l<<in_nbits)-1);
    WordType in_mask  = (1<<in_nbits)-1;
    WordType out_mask = (1<<out_nbits)-1;

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
                for (int k = 0; k < sizeof(WordType)*8; k += out_nbits)
                {
                    int out_cw = threadIdx.x + j;
                    int c = out_cw * out_chans_per_word + k/out_nbits;
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
    cudaStream_t stream)
{
    // Specify thread decomposition (uses up-rounded divisions)
    dim3 tot_block_count((width-1)  / TILE_DIM + 1,
                         (height-1) / TILE_DIM + 1);

    size_t max_grid_dim = round_down_pow2(65536);

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
            transpose_unpack_kernel<dedisp_word><<<grid, block, 0, stream>>>
                (d_in + in_offset,      // in
                 w, h,                  // width, height
                 in_stride, out_stride, // in stride, out stride
                 d_out + out_offset,    // out
                 block_count.x,         // block_count_x
                 block_count.y,         // block_count_y
                 in_nbits,              // in_nbits
                 out_nbits);            // out_nbits
        } // end for block_x_offset
    } // end for block_y_offset
}