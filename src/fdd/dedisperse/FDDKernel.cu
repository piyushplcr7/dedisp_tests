#include <cuda.h>

#include "FDDKernel.hpp"

#include "fdd_kernel.cuh"

#include "common/cuda/CU.h"


/*
 * Helper functions
 */
void FDDKernel::copy_delay_table(
    const void* src,
    size_t count,
    size_t offset,
    cudaStream_t stream)
{
    cu::checkError(cudaMemcpyToSymbolAsync(
        c_delay_table,
        src,
        count, offset,
        cudaMemcpyDeviceToDevice, stream)
    );
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
    cudaStream_t         stream)
{
    // Define thread decomposition
    unsigned grid_x = std::max((int) (ndm / NDM_BATCH_GRID), 1);
    unsigned grid_y = NFREQ_BATCH_GRID;
    dim3 grid(grid_x, grid_y);
    dim3 block(NFREQ_BATCH_BLOCK);

    // Execute the kernel
    #define CALL_KERNEL(NCHAN)        \
    dedisperse_kernel<NCHAN, false>   \
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
        case  16: CALL_KERNEL(16); break;
        case  32: CALL_KERNEL(32); break;
        case  64: CALL_KERNEL(64); break;
        case 128: CALL_KERNEL(128); break;
        case 256: CALL_KERNEL(256); break;
    }
}

/*
 * dedisperse routine
 */
void FDDKernel::scale(
    dedisp_size   height,
    dedisp_size   width,
    dedisp_size   stride,
    dedisp_float* d_data,
    cudaStream_t  stream)
{
    // Define thread decomposition
    dim3 grid(height);
    dim3 block(128);

    // Execute the kernel
    scale_output_kernel<<<grid, block, 0, stream>>>(
        width,
        stride,
        d_data);
}