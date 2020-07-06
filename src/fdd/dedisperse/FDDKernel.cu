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
    unsigned int         ichan_start,
    cudaStream_t         stream)
{
    // Define thread decomposition
    dim3 grid(ndm);
    dim3 block(128);

    // Execute the kernel
    dedisperse_kernel<<<grid, block, 0, stream>>>(
        nfreq,
        nchan,
        dt,
        (float *) d_spin_frequencies,
        (float *) d_dm_list,
        in_stride,
        out_stride,
        (const float2 *) d_in,
        (float2 *) d_out,
        idm_start,
        ichan_start
    );
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