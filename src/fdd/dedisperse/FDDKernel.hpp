#ifndef FDD_KERNEL_H_INCLUDE_GUARD
#define FDD_KERNEL_H_INCLUDE_GUARD

#include <cuda.h>

#include "common/dedisp_types.h"

class FDDKernel
{
    public:
        void copy_delay_table(
            const void* src,
            size_t count,
            size_t offset,
            cudaStream_t stream);

        void launch(
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
            cudaStream_t         stream);

        void scale(
            dedisp_size   height,
            dedisp_size   width,
            dedisp_size   stride,
            dedisp_float  scale,
            dedisp_float* d_data,
            cudaStream_t  stream);

};

#endif // FDD_KERNEL_H_INCLUDE_GUARD