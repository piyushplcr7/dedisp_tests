#include <cmath>
#include <cstdio>
#include <vector>

#include "common/helper.h"

namespace dedisp
{

struct Chunk
{
    // Time domain
    unsigned int isamp_start; // scalar index
    unsigned int isamp_end;   // scalar index

    // Frequency domain
    unsigned int ifreq_start; // scalar index
    unsigned int ifreq_end;   // scalar index
    unsigned int nfreq_good;  // number of good samples
};

void compute_chunks(
    unsigned int nsamp,
    unsigned int nsamp_good,
    unsigned int nfft,
    unsigned int nfreq_chunk_padded,
    unsigned int& nfreq_computed,
    std::vector<Chunk>& chunks)
{
    nfreq_computed = 0;

    for (unsigned int ichunk = 0; ichunk < chunks.size(); ichunk++)
    {
        Chunk& chunk = chunks[ichunk];

        // Compute start and end indices in the time domain
        unsigned int isamp_start = ichunk * nsamp_good;
        unsigned int isamp_end   = isamp_start + nfft;
                     isamp_end   = std::min(isamp_end, nsamp);

        // Compute number of scalar samples in the time domain
        unsigned int nsamp = isamp_end - isamp_start;

        // Store time domain parameters
        chunk.isamp_start = isamp_start;
        chunk.isamp_end   = isamp_end;

        // Compute the number of complex samples in the frequency domain
        unsigned int nfreq       = std::ceil(nsamp / 2.0) + 1;
        unsigned int ifreq_start = ichunk * nfreq_chunk_padded;
        unsigned int ifreq_end   = ifreq_start + nfreq;
        chunk.nfreq_good         = nfreq;
        chunk.ifreq_start        = ifreq_start; // complex index
        chunk.ifreq_end          = ifreq_end;   // complex index

        // Number of good spin frequencies computed
        nfreq_computed += ifreq_end - ifreq_start;
    }
}

void print_chunks(
    std::vector<Chunk>& chunks)
{
    for (unsigned int ichunk = 0; ichunk < chunks.size(); ichunk++)
    {
        Chunk& chunk = chunks[ichunk];
        unsigned int nsamp = chunk.isamp_end - chunk.isamp_start;
        printf("Chunk %d: isamp %5d - %5d, nsamp: %5d, ifreq: %5d - %5d, nfreq: %d\n",
            ichunk, chunk.isamp_start, chunk.isamp_end, nsamp,
            chunk.ifreq_start*2, chunk.ifreq_end*2, chunk.nfreq_good*2);
    }
}

void copy_chunk_output(
    float* src,
    float* dst,
    unsigned int ndm,
    unsigned int nsamp,
    unsigned int nsamp_computed,
    unsigned int nsamp_padded,
    unsigned int nsamp_good,
    std::vector<Chunk>& chunks)
{
    size_t ostart = 0;
    for (auto& chunk : chunks)
    {
        size_t istart = chunk.ifreq_start*2;
        size_t oend   = std::min((size_t) nsamp_computed, ostart + nsamp_good);
        size_t nsamp  = oend - ostart;
        if (nsamp > 0)
        {
            auto *src_ptr = &src[istart];
            auto *dst_ptr = &dst[ostart];
            memcpy2D(
                dst_ptr,                        // dstPtr
                nsamp_computed * sizeof(float), // dstWidth
                src_ptr,                        // srcPtr
                nsamp_padded * sizeof(float),   // srcWidth
                nsamp * sizeof(float),          // widthBytes
                ndm);
            ostart += nsamp;
        }
    }
}

} // end namespace dedisp