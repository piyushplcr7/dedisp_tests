// This value is set according to the constant memory size
// for all NVIDIA GPUs to date, which is 64 KB and
// sizeof(dedisp_float) = 4
#define DEDISP_MAX_NCHANS 16384

// Constant reference for input data
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];

// The number of DMs computed by a single thread block
#define UNROLL_NDM 8

/*
 * Helper functions
 */
inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}

inline __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}


/*
 * dedisperse kernel
 */
template<unsigned int NCHAN>
__global__
void dedisperse_kernel(
    size_t        nfreq,
    float         dt,
    const float*  d_spin_frequencies,
    const float*  d_dm_list,
    size_t in_stride,
    size_t out_stride,
    const float2* d_in,
          float2* d_out,
    unsigned int  idm_start,
    unsigned int  ichan_start)
{
    // The DM that the current block processes
    unsigned int idm_block = blockIdx.x;
    unsigned int idm_offset = gridDim.x;

    // Frequency offset for the current block
    unsigned int ifreq_block = blockIdx.y * blockDim.x;
    unsigned int ifreq_increment = gridDim.y * blockDim.x;
    unsigned int ifreq = ifreq_block + threadIdx.x;

    // Load DMs
    float dms[UNROLL_NDM];
    for (unsigned int i = 0; i < UNROLL_NDM; i++)
    {
        unsigned int idm = idm_start + idm_block + (i * idm_offset);
        dms[i] = d_dm_list[idm];
    }

    for (; ifreq < nfreq; ifreq += ifreq_increment)
    {
        // Load output samples
        float2 sums[UNROLL_NDM];
        #pragma unroll
        for (unsigned int i = 0; i < UNROLL_NDM; i++)
        {
            unsigned int idm = idm_block + i * idm_offset;
            size_t out_idx = idm * out_stride + ifreq;
            sums[i] = d_out[out_idx];
        }

        // Load spin frequency
        float f = d_spin_frequencies[ifreq];

        // Add to output sample
        for (unsigned int ichan = 0; ichan < NCHAN; ichan++)
        {
            // Load input sample
            size_t in_idx = ichan * in_stride + ifreq;
            float2 sample = d_in[in_idx];

            #pragma unroll
            for (unsigned int i = 0; i < UNROLL_NDM; i++)
            {
                // Compute DM delay
                float tdm = dms[i] * c_delay_table[ichan_start + ichan] * dt;

                // Compute phase
                float phase = 2.0f * ((float) M_PI) * f * tdm;

                // Compute phasor
                float2 phasor = make_float2(cosf(phase), sinf(phase));

                // Complex multiply add
                sums[i] += sample * phasor;
            }
        } // end for ichan

        // Store result
        #pragma unroll
        for (unsigned int i = 0; i < UNROLL_NDM; i++)
        {
            unsigned int idm = idm_block + i * idm_offset;
            size_t out_idx = idm * out_stride + ifreq;
            d_out[out_idx] = sums[i];
        }
    } // end if ifreq
} // end dedisperse_kernel


/*
 * scale kernel
 */
__global__
void scale_output_kernel(
    size_t n,
    size_t stride,
    float *d_data)
{
    float scale = 1.0 / n;

    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
    {
        d_data[blockIdx.x * stride + i] *= scale;
    }
}