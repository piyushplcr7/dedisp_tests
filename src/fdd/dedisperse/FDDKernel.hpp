// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef FDD_KERNEL_H_INCLUDE_GUARD
#define FDD_KERNEL_H_INCLUDE_GUARD

#include "gpu_runtime.hpp"

#include "common/dedisp_types.h"

class FDDKernel
{
    public:
        void copy_delay_table(
            const void* src,
            size_t count,
            size_t offset,
            gpuStream_t stream);

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
            gpuStream_t         stream);

        void scale(
            dedisp_size   height,
            dedisp_size   width,
            dedisp_size   stride,
            dedisp_float  scale,
            dedisp_float* d_data,
            gpuStream_t  stream);

};

#endif // FDD_KERNEL_H_INCLUDE_GUARD