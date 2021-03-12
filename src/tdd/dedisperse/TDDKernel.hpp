#ifndef DEDISP_KERNEL_H_INCLUDE_GUARD
#define DEDISP_KERNEL_H_INCLUDE_GUARD

#include <cuda.h>

#include "common/dedisp_types.h"
//#include "common/dedisp_error.hpp"

class DedispKernel
{
    public:
        void copy_delay_table(
            const void* src,
            size_t count,
            size_t offset,
            cudaStream_t stream);

        void copy_killmask(
            const void* src,
            size_t count,
            size_t offset,
            cudaStream_t stream);

        unsigned int get_nsamps_per_thread();

        void use_texture_memory(bool enabled) { m_use_texture_mem = enabled; };

        void launch(
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
            cudaStream_t        stream = 0);


    private:
        /* m_use_texture_mem is a tuning parameter.
        *  Set m_use_texture_mem to true to enable usage of texture memory in the TDDKernel.
        *  Perfromance of the kernel with/without texture memory differs per GPU architecture.
        *  Usage of texture memory is beneficial for Nvidia Pascal,
        *  but causes a performance degradation with Nvidia Turing architecture.
        */
        bool m_use_texture_mem = false;
        dedisp_word* m_d_in = nullptr;
};


#endif // DEDISP_KERNEL_H_INCLUDE_GUARD