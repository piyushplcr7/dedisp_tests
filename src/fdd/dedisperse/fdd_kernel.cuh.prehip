// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later

// Constant reference for input data
// This value is set according to the constant memory size
// for all NVIDIA GPUs to date, which is 64 KB and sizeof(dedisp_float) = 4
#define DEDISP_MAX_NCHANS 16384
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];

// The number of DMs computed by a single thread block
#define NDM_BATCH_GRID 8

// The factor with which the loop over observing channels is unrolled
#define NCHAN_BATCH_THREAD 8

// The spin frequencies are processed in batches of NFREQ_BATCH_GRID thread blocks at once,
// where each thread block processes NFREQ_BATCH_BLOCK spin frequencies per iteration.
#define NFREQ_BATCH_GRID  128
#define NFREQ_BATCH_BLOCK 256

// Option to enable/disable caching input samples in shared memory
#define USE_SHARED_MEMORY 0

/*
 * Helper functions
 */

 // Multiply two float2 operands
inline __device__ float2 operator*(float2 a, float2 b) {
    float2 c;
    asm ("mul.f32 %0,%1,%2;" : "=f"(c.x) : "f"(a.x), "f"(b.x));
    asm ("mul.f32 %0,%1,%2;" : "=f"(c.y) : "f"(a.x), "f"(b.y));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(c.x) : "f"(-a.y), "f"(b.y), "f"(c.x));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(c.y) : "f"( a.y), "f"(b.x), "f"(c.y));
    return c;
}

// Add and assign two float2 operands
inline __device__ void operator+=(float2 &a, float2 b) {
    a.x += b.x;
    a.y += b.y;
}

// Multiply and assign two float2 operands
inline __device__ void operator*=(float2 &a, float2 b) {
    float2 c = a * b;
    a.x = c.x;
    a.y = c.y;
}

// Multiply-and-accumulate (MAC) for complex operands
inline __device__ void cmac(float2 &a, float2 b, float2 c)
{
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.x) : "f"(b.x), "f"(c.x), "f"(a.x));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.y) : "f"(b.x), "f"(c.y), "f"(a.y));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
    asm ("fma.rn.ftz.f32 %0,%1,%2,%3;" : "=f"(a.y) : "f"(b.y), "f"(c.x), "f"(a.y));
}

// Use the Special Function Unit (SFU) for the sine evaluation
inline __device__ float raw_sin(float a)
{
    float r;
    asm ("sin.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
    return r;
}

// Use the Special Function Unit (SFU) for the cosine evaluation
inline __device__ float raw_cos(float a)
{
    float r;
    asm ("cos.approx.ftz.f32 %0,%1;" : "=f"(r) : "f"(a));
    return r;
}

/*
 * dedisperse kernel
 * FDD computes dedispersion as phase rotations in the Fourier domain
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
    // The number of DMs to process
    unsigned int ndm = idm_end - idm_start;

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
        if (idm_idx < ndm)
        {
            dms[i] = idm_idx < idm_end ? d_dm_list[idm_start + idm_idx] : 0.0f;
        }
    }

    // Two input samples (two subsequent spin frequencies, float2 values) are stored as a float4 value
    #if USE_SHARED_MEMORY
    __shared__ float4 s_temp[NCHAN_BATCH_THREAD][NFREQ_BATCH_BLOCK+1];
    #endif

    for (unsigned int ifreq_current = ifreq_start + threadIdx.x; ifreq_current < nfreq; ifreq_current += ifreq_offset)
    {
        // Load output samples
        float2 sums[NDM_BATCH_GRID];
        #pragma unroll
        for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
        {
            unsigned int idm_idx = idm_current + (i * idm_offset);
            if (idm_idx < ndm)
            {
                size_t out_idx = idm_idx * out_stride + ifreq_current;
                sums[i] = d_out[out_idx];
            } else {
                sums[i] = make_float2(0, 0);
            }
        }

        // Load spin frequency
        float f = 0.0f;
        if (ifreq_current < nfreq)
        {
            f = d_spin_frequencies[ifreq_current];
        }

        // Apply phase rotation to input sample and add to output sample
        for (unsigned int ichan_outer = 0; ichan_outer < NCHAN; ichan_outer += NCHAN_BATCH_THREAD)
        {
            // Load samples from device memory to shared memory
            #if USE_SHARED_MEMORY
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
            #endif

            if (extrapolate)
            {   // This is an experimental optimization feature,
                // where extrapolation is used in the computation of the phasors
                // in order to reach a better balance in sin and cos operations vs multiply and accumulate operations.
                // This feature should be further explored to determine whether functional correctness is achieved at all times.

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
                    phasors[i]       = make_float2(raw_cos(phase0), raw_sin(phase0));
                    phasors_delta[i] = make_float2(raw_cos(phase_delta), raw_sin(phase_delta));
                }

                #pragma unroll
                for (unsigned int ichan_inner = 0; ichan_inner < NCHAN_BATCH_THREAD; ichan_inner++)
                {
                    unsigned int ichan = ichan_outer + ichan_inner;

                    // Load input sample
                    #if USE_SHARED_MEMORY
                    float2 sample = ((float2 *) &s_temp[ichan_inner][threadIdx.x/2])[threadIdx.x % 2];
                    #else
                    float2 sample = d_in[ichan * in_stride + ifreq_current];
                    #endif

                    #pragma unroll
                    for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
                    {
                        // Update sum
                        cmac(sums[i], sample, phasors[i]);

                        // Update phasor
                        phasors[i] *= phasors_delta[i];
                    }
                } // end for ichan_inner
            }
            else // Not using the extrapolation feature
            {
                for (unsigned int ichan_inner = 0; ichan_inner < NCHAN_BATCH_THREAD; ichan_inner++)
                {
                    unsigned int ichan = ichan_outer + ichan_inner;

                    // Load input sample
                    #if USE_SHARED_MEMORY
                    float2 sample = ((float2 *) &s_temp[ichan_inner][threadIdx.x/2])[threadIdx.x % 2];
                    #else
                    float2 sample = d_in[ichan * in_stride + ifreq_current];
                    #endif

                    #pragma unroll
                    for (unsigned int i = 0; i < NDM_BATCH_GRID; i++)
                    {
                        // Compute DM delay
                        float tdm = dms[i] * c_delay_table[ichan_start + ichan];

                        // Compute phase
                        float phase = 2.0f * ((float) M_PI) * f * dt * (int)tdm;

                        // Compute phasor
                        float2 phasor = make_float2(raw_cos(phase), raw_sin(phase));

                        // Update sum
                        cmac(sums[i], sample, phasor);
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
                unsigned int idm_idx = idm_current + i * idm_offset;
                if (idm_idx < ndm)
                {
                    size_t out_idx = idm_idx * out_stride + ifreq_current;
                    d_out[out_idx] = sums[i];
                }
            }
        } // end if ifreq_current < nfreq
    } // end for ifreq_current loop
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