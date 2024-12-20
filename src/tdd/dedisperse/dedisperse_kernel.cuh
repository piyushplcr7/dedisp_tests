#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>

/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* Time Domain Dedispersion (TDD)
* is an optimized version of the original dedisp implementation.
*/
// Texture reference for input data
//texture<dedisp_word, 1, cudaReadModeElementType> t_in;

// This value is set according to the constant memory size
// for all NVIDIA GPUs to date, which is 64 KB and
// sizeof(dedisp_float) = 4, sizeof(dedisp_bool) == 4
#define DEDISP_MAX_NCHANS 8192

// Constant reference for input data
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
__constant__ dedisp_bool  c_killmask[DEDISP_MAX_NCHANS];

// Kernel tuning parameters
#define DEDISP_SAMPS_PER_THREAD 2

/*
 * Helper functions
 */
template<int NBITS, typename T=unsigned int>
struct max_value {
    static const T value = (((unsigned)1<<(NBITS-1))-1)*2+1;
};
// Summation type metafunction
template<int IN_NBITS> struct SumType { typedef dedisp_word type; };
// Note: For 32-bit input, we must accumulate using a larger data type
template<> struct SumType<32> { typedef unsigned long long type; };

template<int NBITS, typename T>
inline __host__ __device__
T extract_subword(T value, int idx) {
    enum { MASK = max_value<NBITS,T>::value };
    return (value >> (idx*NBITS)) & MASK;
}

template<int IN_NBITS, typename T, typename SumType>
inline __host__ __device__
T scale_output(SumType sum, dedisp_size nchans) {
    enum { BITS_PER_BYTE = 8 };

    float in_range  = max_value<IN_NBITS>::value;

    // Note: We use floats when out_nbits == 32, and scale to a range of [0:1]
    float out_range = (sizeof(T)==4) ? 1.f
                                     : max_value<sizeof(T)*BITS_PER_BYTE>::value;

    // Note: This emulates what dedisperse_all does for 2-bit HTRU data --> 8-bit
    //         (and will adapt linearly to changes in in/out_nbits or nchans)
    float factor = (3.f * 1024.f) / 255.f / 16.f;
    float scaled = (float)sum * out_range / (in_range * nchans) * factor;
    // Clip to range when necessary
    scaled = (sizeof(T)==4) ? scaled
                            : min(max(scaled, 0.), out_range);
    return (T)scaled;
}

template<int IN_NBITS, typename SumType>
inline __host__ __device__
void set_out_val(dedisp_byte* d_out,
                 dedisp_size out_nbits,
                 dedisp_size idx,
                 SumType sum,
                 dedisp_size nchans) {
    switch( out_nbits ) {
        case 8:
            ((unsigned char*)d_out)[idx] = scale_output<IN_NBITS, unsigned char>(sum, nchans);
            break;
        case 16:
            ((unsigned short*)d_out)[idx] = scale_output<IN_NBITS, unsigned short>(sum, nchans);
            break;
        case 32:
            ((float*)d_out)[idx] = scale_output<IN_NBITS, float>(sum, nchans);
            break;
        default:
            // Error
            break;
    };
}


/*
 * dedisperse kernel
 */

// Note: This assumes consecutive input words are consecutive times,
//         but that consecutive subwords are consecutive channels.
//       E.g., Words bracketed: (t0c0,t0c1,t0c2,t0c3), (t1c0,t1c1,t1c2,t1c3),...
// Note: out_stride should be in units of samples
template<int IN_NBITS, int SAMPS_PER_THREAD,
         int BLOCK_DIM_X, int BLOCK_DIM_Y,
         bool USE_TEXTURE_MEM>
__global__
void dedisperse_kernel(const dedisp_word*  d_in,
                       dedisp_size         nsamps,
                       dedisp_size         nsamps_reduced,
                       dedisp_size         nsamp_blocks,
                       dedisp_size         stride,
                       dedisp_size         dm_count,
                       dedisp_size         dm_stride,
                       dedisp_size         ndm_blocks,
                       dedisp_size         nchans,
                       dedisp_size         chan_stride,
                       dedisp_byte*        d_out,
                       dedisp_size         out_nbits,
                       dedisp_size         out_stride,
                       const dedisp_float* d_dm_list,
                       cudaTextureObject_t t_in)
{
    // Compute compile-time constants
    enum {
        BITS_PER_BYTE  = 8,
        CHANS_PER_WORD = sizeof(dedisp_word) * BITS_PER_BYTE / IN_NBITS
    };

    // Compute the thread decomposition
    dedisp_size samp_block    = blockIdx.x;
    dedisp_size dm_block      = blockIdx.y % ndm_blocks;

    dedisp_size samp_idx      = samp_block   * BLOCK_DIM_X + threadIdx.x;
    dedisp_size dm_idx        = dm_block     * BLOCK_DIM_Y + threadIdx.y;
    dedisp_size nsamp_threads = nsamp_blocks * BLOCK_DIM_X;

    dedisp_size ndm_threads   = ndm_blocks * BLOCK_DIM_Y;

    // Iterate over grids of DMs
    for( ; dm_idx < dm_count; dm_idx += ndm_threads ) {

    // Look up the dispersion measure
    dedisp_float dm = d_dm_list[dm_idx*dm_stride];

    // Loop over samples
    for( ; samp_idx < nsamps_reduced; samp_idx += nsamp_threads ) {
        typedef typename SumType<IN_NBITS>::type sum_type;
        sum_type sum[SAMPS_PER_THREAD];

        #pragma unroll
        for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
            sum[s] = 0;
        }

        // Loop over channel words
        for( dedisp_size chan_word=0; chan_word<nchans;
             chan_word+=CHANS_PER_WORD ) {
            // Pre-compute the memory offset
            dedisp_size offset =
                samp_idx*SAMPS_PER_THREAD
                + chan_word/CHANS_PER_WORD * stride;

            // Loop over channel subwords
            for( dedisp_size chan_sub=0; chan_sub<CHANS_PER_WORD; ++chan_sub ) {
                dedisp_size chan_idx = (chan_word + chan_sub)*chan_stride;

                // Look up the fractional delay
                dedisp_float frac_delay = c_delay_table[chan_idx];
                // Compute the integer delay
                dedisp_size delay = __float2uint_rn(dm * frac_delay);

                // Loop over samples per thread
                // Note: Unrolled to ensure the sum[] array is stored in regs
                #pragma unroll
                for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                    // Grab the word containing the sample
                    dedisp_word sample = USE_TEXTURE_MEM
                        ? tex1Dfetch<dedisp_word>(t_in, offset+s + delay)
                        : d_in[offset + s + delay];

                    // Extract the desired subword and accumulate
                    sum[s] +=
                        c_killmask[chan_idx]*
                        extract_subword<IN_NBITS>(sample,chan_sub);
                }
            }
        }

        // Write sums to global mem
        dedisp_size out_idx = ( samp_idx*SAMPS_PER_THREAD +
                                dm_idx * out_stride);
        #pragma unroll
        for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
            if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
                set_out_val<IN_NBITS>(d_out, out_nbits,
                                      out_idx + s,
                                      sum[s], nchans);
        }

    } // End of sample loop

    } // End of DM loop
}