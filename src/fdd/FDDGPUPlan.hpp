// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef H_FDD_GPU_PLAN_INCLUDE_GUARD
#define H_FDD_GPU_PLAN_INCLUDE_GUARD

//#define EXPORT_DEDISP_TIME_SERIES

#include "GPUPlan.hpp"
#include "FDDCPUPlan.hpp"

namespace dedisp
{

class FDDGPUPlan : public GPUPlan {

public:
    // Constructor
    FDDGPUPlan(
        size_type  nchans,
        float_type dt,
        float_type f0,
        float_type df,
        int device_idx = 0);

    // Destructor
    ~FDDGPUPlan();

    // Public interface for FDD on GPU
    virtual void execute(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits,
        unsigned         flags = 0) override;

private:
    // Private interface for FDD on GPU
    virtual void execute_gpu(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    // Private interface for FDD on GPU with time segmentation of input data
    virtual void execute_gpu_segmented(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    // Helper method to generate a spin frequency table
    void generate_spin_frequency_table(
        dedisp_size nfreq,
        dedisp_size nsamp,
        dedisp_float dt);

    // Host arrays
    std::vector<dedisp_float> h_spin_frequencies; // size = nfreq
    std::vector<cu::HostMemory> h_data_t_nu_;
    std::vector<cu::HostMemory> h_data_t_dm_;

    // Device arrays
    cu::DeviceMemory d_spin_frequencies; // type = dedisp_float
    std::vector<cu::DeviceMemory> d_data_t_nu_;
    std::vector<cu::DeviceMemory> d_data_x_dm_;
};

} // end namespace dedisp

#endif // H_FDD_GPU_PLAN_INCLUDE_GUARD