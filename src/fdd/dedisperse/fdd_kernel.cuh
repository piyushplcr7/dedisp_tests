// This value is set according to the constant memory size
// for all NVIDIA GPUs to date, which is 64 KB and
// sizeof(dedisp_float) = 4
#define DEDISP_MAX_NCHANS 16384

// Number of threads in the x dimension of a thread block
#define BLOCK_DIM_X 128

// Constant reference for input data
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];

// The number of DMs computed by a single thread block
#define UNROLL_NDM 8

// The factor with which the loop over observing channels is unrolled
#define NCHAN_BATCH 8

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

    // Load DMs
    float dms[UNROLL_NDM];
    for (unsigned int i = 0; i < UNROLL_NDM; i++)
    {
        unsigned int idm = idm_start + idm_block + (i * idm_offset);
        dms[i] = d_dm_list[idm];
    }

    // Shared memory
    __shared__ float4 s_temp[NCHAN_BATCH][BLOCK_DIM_X/2];

    for (unsigned int ifreq_outer = ifreq_block; ifreq_outer < nfreq; ifreq_outer += ifreq_increment)
    {
        unsigned int ifreq_inner = threadIdx.x;
        unsigned int ifreq = ifreq_outer + ifreq_inner;

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
        float f;
        if (ifreq < nfreq)
        {
            f = d_spin_frequencies[ifreq];
        }

        // Add to output sample
        for (unsigned int ichan_outer = 0; ichan_outer < NCHAN; ichan_outer += NCHAN_BATCH)
        {
            // Load samples from device memory to shared memory
            __syncthreads();
            for (unsigned int i = threadIdx.x; i < NCHAN_BATCH * (BLOCK_DIM_X/2); i += blockDim.x)
            {
                unsigned int ichan_inner = i / (BLOCK_DIM_X/2);
                unsigned int ifreq_inner = i % (BLOCK_DIM_X/2);
                unsigned int ichan = ichan_outer + ichan_inner;
                size_t in_idx = ichan * in_stride + (ifreq - threadIdx.x);
                float4 *sample_ptr = (float4 *) &d_in[in_idx];
                s_temp[ichan_inner][ifreq_inner] = sample_ptr[ifreq_inner];
            }
            __syncthreads();

            if (extrapolate)
            {
                // Compute initial and delta phasor values
                float2 phasors[UNROLL_NDM];
                float2 phasors_delta[UNROLL_NDM];
                for (unsigned int i = 0; i < UNROLL_NDM; i++)
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
                for (unsigned int ichan_inner = 0; ichan_inner < NCHAN_BATCH; ichan_inner++)
                {
                    // Load input sample
                    float2 sample = ((float2 *) &s_temp[ichan_inner][threadIdx.x/2])[threadIdx.x % 2];

                    #pragma unroll
                    for (unsigned int i = 0; i < UNROLL_NDM; i++)
                    {
                        // Update sum
                        sums[i] += sample * phasors[i];

                        // Update phasor
                        phasors[i] *= phasors_delta[i];
                    }
                } // end for ichan_inner
            } else {
                for (unsigned int ichan_inner = 0; ichan_inner < NCHAN_BATCH; ichan_inner++)
                {
                    // Load input sample
                    unsigned int ichan = ichan_outer + ichan_inner;
                    float2 sample = ((float2 *) &s_temp[ichan_inner][threadIdx.x/2])[threadIdx.x % 2];

                    #pragma unroll
                    for (unsigned int i = 0; i < UNROLL_NDM; i++)
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
        if (ifreq < nfreq)
        {
            #pragma unroll
            for (unsigned int i = 0; i < UNROLL_NDM; i++)
            {
                unsigned int idm = idm_block + i * idm_offset;
                size_t out_idx = idm * out_stride + ifreq;
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
    float *d_data)
{
    float scale = 1.0 / n;

    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
    {
        d_data[blockIdx.x * stride + i] *= scale;
    }
}