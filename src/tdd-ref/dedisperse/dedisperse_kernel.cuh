// Texture reference for input data
texture<dedisp_word, 1, cudaReadModeElementType> t_in;

// Constant reference for input data
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
__constant__ dedisp_bool  c_killmask[DEDISP_MAX_NCHANS];

/*
 * Helper functions
 */
template<int NBITS, typename T=unsigned int>
struct max_value {
    static const T value = (((unsigned)1<<(NBITS-1))-1)*2+1;
};

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
    // This emulates dedisperse_all, but is specific to 8-bit output
    // Note: This also breaks the sub-band algorithm
    //return (dedisp_word)(((unsigned int)sum >> 4) - 128 - 128);
    // HACK
    //return (T)(sum / 16 - 128 - 128);

    // This uses the full range of the output bits
    //return (T)((double)sum / ((double)nchans * max_value<IN_NBITS>::value)
    //		   * max_value<sizeof(T)*BITS_PER_BYTE>::value );
    /*
    // This assumes the input data have mean=range/2 and then scales by
    //   assuming the SNR goes like sqrt(nchans).
    double in_range  = max_value<IN_NBITS>::value;
    double mean      = 0.5 *      (double)nchans  * in_range;
    double max_val   = 0.5 * sqrt((double)nchans) * in_range;

    // TODO: There are problems with the output scaling when in_nbits is small
    //         (e.g., in_nbits < 8). Not sure what to do about it at this stage.

    // TESTING This fixes 1-bit
    // TODO: See test_quantised_rms.py for further exploration of this
    //double max       = 0.5 * sqrt((double)nchans) * in_range * 2*4.545454; // HACK
    // TESTING This fixes 2-bit
    //double max       = 0.5 * sqrt((double)nchans) * in_range * 0.8*4.545454; // HACK
    // TESTING This fixes 4-bit
    //double max       = 0.5 * sqrt((double)nchans) * in_range * 0.28*4.545454; // HACK
    double out_range = max_value<sizeof(T)*BITS_PER_BYTE>::value;
    double out_mean  = 0.5 * out_range;
    double out_max   = 0.5 * out_range;
    double scaled = floor((sum-mean)/max_val * out_max + out_mean + 0.5);
    */
    float in_range  = max_value<IN_NBITS>::value;
    // Note: We use floats when out_nbits == 32, and scale to a range of [0:1]
    float out_range = (sizeof(T)==4) ? 1.f
                                     : max_value<sizeof(T)*BITS_PER_BYTE>::value;
    //float scaled = (float)sum / in_range / sqrt((float)nchans) * out_range;
    //float scaled = (float)sum / (in_range * nchans) * out_range;
    //float scaled = sum * ((float)out_range / in_range / 85.f) / 16.f;

    // Note: This emulates what dedisperse_all does for 2-bit HTRU data --> 8-bit
    //         (and will adapt linearly to changes in in/out_nbits or nchans)
    float factor = (3.f * 1024.f) / 255.f / 16.f;
    float scaled = (float)sum * out_range / (in_range * nchans) * factor;
    // Clip to range when necessary
    scaled = (sizeof(T)==4) ? scaled
                            : min(max(scaled, 0.), out_range);
    return (T)scaled;
}

template<typename T, int IN_NBITS, typename SumType>
inline __host__ __device__
void set_out_val(dedisp_byte* d_out, dedisp_size idx,
                 SumType sum, dedisp_size nchans) {
    ((T*)d_out)[idx] = scale_output<IN_NBITS,T>(sum, nchans);
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
                       dedisp_size         batch_in_stride,
                       dedisp_size         batch_dm_stride,
                       dedisp_size         batch_chan_stride,
                       dedisp_size         batch_out_stride)
{
    // Compute compile-time constants
    enum {
        BITS_PER_BYTE  = 8,
        CHANS_PER_WORD = sizeof(dedisp_word) * BITS_PER_BYTE / IN_NBITS
    };

    // Compute the thread decomposition
    dedisp_size samp_block    = blockIdx.x;
    dedisp_size dm_block      = blockIdx.y % ndm_blocks;
    dedisp_size batch_block   = blockIdx.y / ndm_blocks;

    dedisp_size samp_idx      = samp_block   * BLOCK_DIM_X + threadIdx.x;
    dedisp_size dm_idx        = dm_block     * BLOCK_DIM_Y + threadIdx.y;
    dedisp_size batch_idx     = batch_block;
    dedisp_size nsamp_threads = nsamp_blocks * BLOCK_DIM_X;

    dedisp_size ndm_threads   = ndm_blocks * BLOCK_DIM_Y;

    // Iterate over grids of DMs
    for( ; dm_idx < dm_count; dm_idx += ndm_threads ) {

    // Look up the dispersion measure
    // Note: The dm_stride and batch_dm_stride params are only used for the
    //         sub-band method.
    dedisp_float dm = d_dm_list[dm_idx*dm_stride + batch_idx*batch_dm_stride];

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
                + chan_word/CHANS_PER_WORD * stride
                + batch_idx * batch_in_stride;

            // Loop over channel subwords
            for( dedisp_size chan_sub=0; chan_sub<CHANS_PER_WORD; ++chan_sub ) {
                dedisp_size chan_idx = (chan_word + chan_sub)*chan_stride
                    + batch_idx*batch_chan_stride;

                // Look up the fractional delay
                dedisp_float frac_delay = c_delay_table[chan_idx];
                // Compute the integer delay
                dedisp_size delay = __float2uint_rn(dm * frac_delay);

                if( USE_TEXTURE_MEM ) { // Pre-Fermi path
                    // Loop over samples per thread
                    // Note: Unrolled to ensure the sum[] array is stored in regs
                    #pragma unroll
                    for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                        // Grab the word containing the sample from texture mem
                        dedisp_word sample = tex1Dfetch(t_in, offset+s + delay);

                        // Extract the desired subword and accumulate
                        sum[s] +=
                            // TODO: Pre-Fermi cards are faster with 24-bit mul
                            /*__umul24*/(c_killmask[chan_idx] *//,
                                     extract_subword<IN_NBITS>(sample,chan_sub));
                    }
                }
                else { // Fermi path
                    // Note: Unrolled to ensure the sum[] array is stored in regs
                    #pragma unroll
                    for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                        // Grab the word containing the sample from global mem
                        dedisp_word sample = d_in[offset + s + delay];

                        // Extract the desired subword and accumulate
                        sum[s] +=
                            c_killmask[chan_idx] *
                            extract_subword<IN_NBITS>(sample, chan_sub);
                    }
                }
            }
        }

        // Write sums to global mem
        // Note: This is ugly, but easy, and doesn't hurt performance
        dedisp_size out_idx = ( samp_idx*SAMPS_PER_THREAD +
                                dm_idx * out_stride +
                                batch_idx * batch_out_stride );
        switch( out_nbits ) {
            case 8:
                #pragma unroll
                for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                    if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
                        set_out_val<unsigned char, IN_NBITS>(d_out, out_idx + s,
                                                             sum[s], nchans);
                }
                break;
            case 16:
                #pragma unroll
                for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                    if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
                        set_out_val<unsigned short, IN_NBITS>(d_out, out_idx + s,
                                                              sum[s], nchans);
                }
                break;
            case 32:
                #pragma unroll
                for( dedisp_size s=0; s<SAMPS_PER_THREAD; ++s ) {
                    if( samp_idx*SAMPS_PER_THREAD + s < nsamps )
                        set_out_val<float, IN_NBITS>(d_out, out_idx + s,
                                                     sum[s], nchans);
                }
                break;
            default:
                // Error
                break;
        }

    } // End of sample loop

    } // End of DM loop
}