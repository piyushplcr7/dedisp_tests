// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include "gpu_runtime.hpp"
#include "FDDKernel.hpp"
#include "fdd_kernel.cuh"
#include "common/cuda/CU.h"
#include <cstring>
#include <cmath>

/*
 * Helper functions
 */
void FDDKernel::copy_delay_table(
    const void* src,
    size_t count,
    size_t offset,
    gpuStream_t stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    gpuMemcpyToSymbolAsync(gpuSymbol(
        c_delay_table),
        src,
        count, offset,
        gpuMemcpyDeviceToDevice, stream);
#elif defined(USE_OPENMP)
    std::memcpy((char*)gpuSymbol(c_delay_table) + offset, src, count);
#endif
}

unsigned long div_round_up(unsigned long a, unsigned long b) {
    return (a-1) / b + 1;
}


/*
 * dedisperse routine
 */
void FDDKernel::launch(
    dedisp_size          ndm,
    dedisp_size          nfreq,
    dedisp_size          nchan,
    float                dt,
    const dedisp_float*  d_spin_frequencies,
    const dedisp_float*  d_dm_list,
    const dedisp_float2* d_in,
    const dedisp_float2* d_out,
    dedisp_size          in_stride,
    dedisp_size          out_stride,
    unsigned int         idm_start,
    unsigned int         idm_end,
    unsigned int         ichan_start,
    gpuStream_t         stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    // Define thread decomposition
    unsigned grid_x = std::max((int) ((ndm + NDM_BATCH_GRID) / NDM_BATCH_GRID), 1);
    unsigned grid_y = NFREQ_BATCH_GRID;
    dim3 grid(grid_x, grid_y);
    dim3 block(NFREQ_BATCH_BLOCK);

    /* Execute the kernel
    *  The second kernel argument can be set to true
    *  in order to enable an experimental optimization feature,
    *  where extrapolation is used in the computation of the phasors.
    *  Boudary conditions should be further explored to determine
    *  functional correctness at all times.
    *  Leaving this feature in because it might be beneficial
    *  depending on the system configurations.
    */
    #define CALL_KERNEL(NCHAN)        \
    dedisperse_kernel<NCHAN, false>    \
    <<<grid, block, 0, stream>>>(     \
        nfreq,                        \
        dt,                           \
        (float *) d_spin_frequencies, \
        (float *) d_dm_list,          \
        in_stride,                    \
        out_stride,                   \
        (const float2 *) d_in,        \
        (float2 *) d_out,             \
        idm_start,                    \
        idm_end,                      \
        ichan_start);

    switch (nchan)
    {
        case  1: CALL_KERNEL(1); break;
        case  16: CALL_KERNEL(16); break;
        case  32: CALL_KERNEL(32); break;
        case  64: CALL_KERNEL(64); break;
        case 128: CALL_KERNEL(128); break;
        case 256: CALL_KERNEL(256); break;
    }
#elif defined(USE_OPENMP)
    // Host port of dedisperse_kernel<NCHAN,false>: freq-domain phase rotation,
    // ACCUMULATING onto d_out (read-modify-write per channel gulp).
    const float* in  = reinterpret_cast<const float*>(d_in);
    float*       out = reinterpret_cast<float*>(const_cast<dedisp_float2*>(d_out));
    // idm_idx is 0-based relative to this job (matches kernel's idm_idx = idm_current + i*idm_offset).
    // d_out buffer is zeroed fresh per dm_job before the first channel gulp (dm_job.d_data_x_dm->zero()).
    // d_dm_list is indexed as d_dm_list[idm_start + idm_idx] in the kernel, equivalent to d_dm_list[idm] here.
    #pragma omp parallel for collapse(2)
    for (unsigned idm = idm_start; idm < idm_end; ++idm) {
        for (size_t ifreq = 0; ifreq < nfreq; ++ifreq) {
            float dm = d_dm_list[idm];
            float f  = d_spin_frequencies[ifreq];
            // idm_idx is 0-based: kernel uses idm_idx*out_stride, not (idm_start+idm_idx)*out_stride
            size_t idm_idx = idm - idm_start;
            size_t o = (idm_idx * out_stride + ifreq) * 2;
            float sx = out[o], sy = out[o + 1];
            for (unsigned ichan = 0; ichan < nchan; ++ichan) {
                size_t iidx = ((size_t)ichan * in_stride + ifreq) * 2;
                float bx = in[iidx], by = in[iidx + 1];
                float tdm   = roundf(dm * c_delay_table[ichan_start + ichan]);
                float phase = 2.0f * (float)M_PI * f * tdm;
                float c = cosf(phase), s = sinf(phase);
                sx += bx * c - by * s;   // sum += sample * phasor (complex)
                sy += bx * s + by * c;
            }
            out[o] = sx; out[o + 1] = sy;
        }
    }
#endif
}

/*
 * dedisperse routine
 */
void FDDKernel::scale(
    dedisp_size   height,
    dedisp_size   width,
    dedisp_size   stride,
    dedisp_float  scale,
    dedisp_float* d_data,
    gpuStream_t  stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    // Define thread decomposition
    dim3 grid(height);
    dim3 block(128);

    // Execute the kernel
    scale_output_kernel<<<grid, block, 0, stream>>>(
        width,
        stride,
        scale,
        d_data);
#elif defined(USE_OPENMP)
    #pragma omp parallel for
    for (size_t row = 0; row < height; ++row)
        for (size_t i = 0; i < width; ++i)
            d_data[row * stride + i] *= scale;   // matches scale_output_kernel
#endif
}
