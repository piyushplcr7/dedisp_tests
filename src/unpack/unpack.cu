#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "dedisp_defines.h"
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
    thrust::device_ptr<dedisp_word> d_unpacked_begin(d_unpacked);

    dedisp_size expansion = out_nbits / in_nbits;
    dedisp_size in_count  = nsamps * nchan_words;
    dedisp_size out_count = in_count * expansion;

    using thrust::make_counting_iterator;

    thrust::transform(thrust::cuda::par.on(stream),
                      make_counting_iterator<unsigned int>(0),
                      make_counting_iterator<unsigned int>(out_count),
                      d_unpacked_begin,
                      unpack_functor<dedisp_word>(d_transposed, nsamps,
                                                  in_nbits, out_nbits));
}