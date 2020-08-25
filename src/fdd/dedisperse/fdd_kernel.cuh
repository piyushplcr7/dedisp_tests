// This value is set according to the constant memory size
// for all NVIDIA GPUs to date, which is 64 KB and
// sizeof(dedisp_float) = 4
#define DEDISP_MAX_NCHANS 16384

// Constant reference for input data
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];

// The number of DMs computed by a single thread block
#define NDM_BATCH_GRID 8

// The factor with which the loop over observing channels is unrolled
#define NCHAN_BATCH_THREAD 8

// The spin frequencies are processed in batches of NFREQ_BATCH_GRID thread blocks at once,
// where each thread block processes NFREQ_BATCH_BLOCK spin frequencies per iteration.
#define NFREQ_BATCH_GRID  32
#define NFREQ_BATCH_BLOCK 128

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

inline __device__ void operator*=(float2 &a, float2 b) {
    float2 c = a * b;
    a.x = c.x;
    a.y = c.y;
}

/*
 * dedisperse kernel
 */
template<unsigned int NCHAN, bool extrapolate>
__global__
void dedisperse_kernel(
    size_t        nfreq,
    float         dt,
    const float*  d_spin_frequencies,
    const float*  d_dm_list,
    size_t        in_stride,
    size_t        out_stride,
    const float2* d_in,
          float2* d_out,
    unsigned int  idm_start,
    unsigned int  idm_end,
    unsigned int  ichan_start)
{
    // The DM that the current block processes
    unsigned int idm_current = blockIdx.x;
    // The DM offset is the number of DMs processed by all thread blocks
    unsigned int idm_offset = gridDim.x;

    // The first spin frequency that the current block processes
    unsigned int ifreq_start = blockIdx.y * blockDim.x;
    // The spin frequency offset is the number of spin frequencies processed
    // by all thread blocks (in the y-dimension) and threads (in the x-dimension)
    unsigned int ifreq_offset = gridDim.y * blockDim.x;

    // Load DMs
    float dms[NDM_BATCH_GRID];
    for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
    {
        unsigned int idm_idx = idm_current + (i * idm_offset);
        dms[i] = idm_idx < idm_end ? d_dm_list[idm_start + idm_idx] : 0.0f;
    }

    // Two input samples (two subsequent spin frequencies, float2 values) are stored as a float4 value
    __shared__ float4 s_temp[NCHAN_BATCH_THREAD][NFREQ_BATCH_BLOCK/2];

    for (unsigned int ifreq_current = ifreq_start + threadIdx.x; ifreq_current < nfreq; ifreq_current += ifreq_offset)
    {
        // Load output samples
        float2 sums[NDM_BATCH_GRID];
        #pragma unroll
        for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
        {
            unsigned int idm_idx = idm_current + (i * idm_offset);
            size_t out_idx = idm_idx * out_stride + ifreq_current;
            sums[i] = d_out[out_idx];
        }

        // Load spin frequency
        float f = 0.0f;
        if (ifreq_current < nfreq)
        {
            f = d_spin_frequencies[ifreq_current];
        }

        // Add to output sample
        for (unsigned int ichan_outer = 0; ichan_outer < NCHAN; ichan_outer += NCHAN_BATCH_THREAD)
        {
            // Load samples from device memory to shared memory
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < NCHAN_BATCH_THREAD * (NFREQ_BATCH_BLOCK/2); i += blockDim.x)
            {
                unsigned int ichan_inner = i / (NFREQ_BATCH_BLOCK/2);
                unsigned int ifreq_inner = i % (NFREQ_BATCH_BLOCK/2);
                unsigned int ichan = ichan_outer + ichan_inner;
                size_t in_idx = ichan * in_stride + (ifreq_current - threadIdx.x);
                float4 *sample_ptr = (float4 *) &d_in[in_idx];
                s_temp[ichan_inner][ifreq_inner] = sample_ptr[ifreq_inner];
            }
            __syncthreads();

            if (extrapolate)
            {
                // Compute initial and delta phasor values
                float2 phasors[NDM_BATCH_GRID];
                float2 phasors_delta[NDM_BATCH_GRID];
                for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
                {
                    float tdm0 = dms[i] * c_delay_table[ichan_start + ichan_outer + 0] * dt;
                    float tdm1 = dms[i] * c_delay_table[ichan_start + ichan_outer + 1] * dt;
                    float phase0 = 2.0f * ((float) M_PI) * f * tdm0;
                    float phase1 = 2.0f * ((float) M_PI) * f * tdm1;
                    float phase_delta = phase1 - phase0;
                    phasors[i]       = make_float2(cosf(phase0), sinf(phase0));
                    phasors_delta[i] = make_float2(cosf(phase_delta), sinf(phase_delta));
                }

                #pragma unroll
                for (unsigned int ichan_inner = 0; ichan_inner < NCHAN_BATCH_THREAD; ichan_inner++)
                {
                    // Load input sample
                    float2 sample = ((float2 *) &s_temp[ichan_inner][threadIdx.x/2])[threadIdx.x % 2];

                    #pragma unroll
                    for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
                    {
                        // Update sum
                        sums[i] += sample * phasors[i];

                        // Update phasor
                        phasors[i] *= phasors_delta[i];
                    }
                } // end for ichan_inner
            } else {
                for (unsigned int ichan_inner = 0; ichan_inner < NCHAN_BATCH_THREAD; ichan_inner++)
                {
                    // Load input sample
                    unsigned int ichan = ichan_outer + ichan_inner;
                    float2 sample = ((float2 *) &s_temp[ichan_inner][threadIdx.x/2])[threadIdx.x % 2];

                    #pragma unroll
                    for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
                    {
                        // Compute DM delay
                        float tdm = dms[i] * c_delay_table[ichan_start + ichan] * dt;

                        // Compute phase
                        float phase = 2.0f * ((float) M_PI) * f * tdm;

                        // Compute phasor
                        float2 phasor = make_float2(cosf(phase), sinf(phase));

                        // Update sum
                        sums[i] += sample * phasor;
                    }
                } // end for ichan_inner
            } // end if extrapolate
        } // end for ichan_outer

        // Store result
        if (ifreq_current < nfreq)
        {
            #pragma unroll
            for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
            {
                unsigned int idm = idm_current + i * idm_offset;
                size_t out_idx = idm * out_stride + ifreq_current;
                d_out[out_idx] = sums[i];
            }
        } // end if ifreq
    } // end for ifreq_outer
} // end dedisperse_kernel


/*
 * scale kernel
 */
__global__
void scale_output_kernel(
    size_t n,
    size_t stride,
    float scale,
    float *d_data)
{
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
    {
        d_data[blockIdx.x * stride + i] *= scale;
    }
}