/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* Time Domain Dedispersion (TDD)
* is an optimized version of the original dedisp implementation.
*/
#include "TDDKernel.hpp"
#include "dedisperse_kernel.cuh"
#include "common/cuda/CU.h"
#include "common/dedisp_error.hpp"

// Kernel tuning parameters
#define DEDISP_BLOCK_SIZE       256
#define DEDISP_BLOCK_SAMPS      8

/*
 * Helper functions
 */
void DedispKernel::copy_delay_table(
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

void DedispKernel::copy_killmask(
    const void* src,
    size_t count,
    size_t offset,
    cudaStream_t stream)
{
    cu::checkError(cudaMemcpyToSymbolAsync(
        c_killmask,
        src,
        count, offset,
        cudaMemcpyDeviceToDevice, stream)
    );
}

unsigned int DedispKernel::get_nsamps_per_thread()
{
    return DEDISP_SAMPS_PER_THREAD;
}

/*
 * dedisperse routine
 */
void DedispKernel::launch(
    const dedisp_word*  d_in,
    dedisp_size         in_stride,
    dedisp_size         nsamps,
    dedisp_size         in_nbits,
    dedisp_size         nchans,
    dedisp_size         chan_stride,
    const dedisp_float* d_dm_list,
    dedisp_size         dm_count,
    dedisp_size         dm_stride,
    dedisp_byte*        d_out,
    dedisp_size         out_stride,
    dedisp_size         out_nbits,
    cudaStream_t        stream)
{
    enum {
        BITS_PER_BYTE            = 8,
        BYTES_PER_WORD           = sizeof(dedisp_word) / sizeof(dedisp_byte),
        BLOCK_DIM_X              = DEDISP_BLOCK_SAMPS,
        BLOCK_DIM_Y              = DEDISP_BLOCK_SIZE / DEDISP_BLOCK_SAMPS,
        MAX_CUDA_GRID_SIZE_X     = 65535,
        MAX_CUDA_GRID_SIZE_Y     = 65535,
        MAX_CUDA_1D_TEXTURE_SIZE = (1<<27)
    };

    // Create texture object
    cudaTextureObject_t t_in = 0;

    // Initialise texture memory if necessary
    // --------------------------------------
    if (m_use_texture_mem && d_in != m_d_in)
    {
        m_d_in = (dedisp_word *) d_in;

        dedisp_size chans_per_word = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;
        dedisp_size nchan_words    = nchans / chans_per_word;
        dedisp_size input_words    = in_stride * nchan_words;

        // Check the texture size limit
        if( input_words > MAX_CUDA_1D_TEXTURE_SIZE ) {
            throw_error(DEDISP_INTERNAL_GPU_ERROR);
        }

        // Still usable in later cuda versions
        cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<dedisp_word>();
        // Need to use cudaArray_t based on the example in Nvidia doc
        cudaArray_t cuArray;
        // Allocating memory for cuArray
        cudaMallocArray(&cuArray, &channel_desc, input_words, 1);

        // Copy data from d_in to CUDA array. Redundant, should be improved!
        //cudaMemcpyToArray(cuArray, 0, 0, d_in, input_words * sizeof(dedisp_word), cudaMemcpyDeviceToDevice);
        cudaMemcpy2DToArray(cuArray, 0, 0, d_in, input_words * sizeof(dedisp_word), input_words * sizeof(dedisp_word), 1, cudaMemcpyDeviceToDevice);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc)); // Clear the structure
        texDesc.addressMode[0] = cudaAddressModeClamp; // Specify address mode (clamp, wrap, etc.)
        texDesc.addressMode[1] = cudaAddressModeClamp; // Only for 2D, set 1D to clamp
        texDesc.filterMode = cudaFilterModePoint; // Set filter mode (linear, point, etc.)
        texDesc.readMode = cudaReadModeElementType; // Set read mode (element type)
        texDesc.normalizedCoords = 0; // Use unnormalized coordinates (1D texture)

        // Create texture object
        //cudaTextureObject_t t_in = 0;
        cudaCreateTextureObject(&t_in, &resDesc, &texDesc, NULL);

        // Bind the texture memory
        /* cu::checkError(cudaBindTexture(
            0, t_in, d_in, channel_desc,
            input_words * sizeof(dedisp_word))
        ); */
    }
    // --------------------------------------

    // Define thread decomposition
    // Note: Block dimensions x and y represent time samples and DMs respectively
    dim3 block(BLOCK_DIM_X,
               BLOCK_DIM_Y);
    // Note: Grid dimension x represents time samples. Dimension y represents DMs

    // Divide and round up
    dedisp_size nsamp_blocks = (nsamps - 1)
        / ((dedisp_size)DEDISP_SAMPS_PER_THREAD*block.x) + 1;
    dedisp_size ndm_blocks   = (dm_count - 1) / (dedisp_size)block.y + 1;

    // Constrain the grid size to the maximum allowed
    ndm_blocks = min((unsigned int)ndm_blocks,
                     (unsigned int)(MAX_CUDA_GRID_SIZE_Y));

    dim3 grid(nsamp_blocks, ndm_blocks);

    // Divide and round up
    dedisp_size nsamps_reduced = (nsamps - 1) / DEDISP_SAMPS_PER_THREAD + 1;

    // Execute the kernel
#define DEDISP_CALL_KERNEL(NBITS, USE_TEXTURE_MEM)						\
    dedisperse_kernel<NBITS,DEDISP_SAMPS_PER_THREAD,BLOCK_DIM_X,        \
                      BLOCK_DIM_Y,USE_TEXTURE_MEM>                      \
        <<<grid, block, 0, stream>>>(d_in,								\
                                     nsamps,							\
                                     nsamps_reduced,					\
                                     nsamp_blocks,						\
                                     in_stride,							\
                                     dm_count,							\
                                     dm_stride,							\
                                     ndm_blocks,						\
                                     nchans,							\
                                     chan_stride,						\
                                     d_out,								\
                                     out_nbits,							\
                                     out_stride,						\
                                     d_dm_list,                         \
                                     t_in)
    // Note: Here we dispatch dynamically on nbits for supported values
    if( m_use_texture_mem ) {
        switch( in_nbits ) {
            case 1:  DEDISP_CALL_KERNEL(1,true);  break;
            case 2:  DEDISP_CALL_KERNEL(2,true);  break;
            case 4:  DEDISP_CALL_KERNEL(4,true);  break;
            case 8:  DEDISP_CALL_KERNEL(8,true);  break;
            case 16: DEDISP_CALL_KERNEL(16,true); break;
            case 32: DEDISP_CALL_KERNEL(32,true); break;
            default: /* should never be reached */ break;
        }
    }
    else {
        switch( in_nbits ) {
            case 1:  DEDISP_CALL_KERNEL(1,false);  break;
            case 2:  DEDISP_CALL_KERNEL(2,false);  break;
            case 4:  DEDISP_CALL_KERNEL(4,false);  break;
            case 8:  DEDISP_CALL_KERNEL(8,false);  break;
            case 16: DEDISP_CALL_KERNEL(16,false); break;
            case 32: DEDISP_CALL_KERNEL(32,false); break;
            default: /* should never be reached */ break;
        }
    }
#undef DEDISP_CALL_KERNEL
}