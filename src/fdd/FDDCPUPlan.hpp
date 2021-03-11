// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef H_FDD_CPU_PLAN_INCLUDE_GUARD
#define H_FDD_CPU_PLAN_INCLUDE_GUARD

#include <vector>

#include "Plan.hpp"

namespace dedisp
{

class FDDCPUPlan : public Plan {
friend class FDDGPUPlan;

public:
    // Constructor
    FDDCPUPlan(
        size_type  nchans,
        float_type dt,
        float_type f0,
        float_type df,
        int device_index = 0); //dummy parameter to match FDDGPUPlan constructor
        //device_index might be use later to select the numa node

    // Destructor
    ~FDDCPUPlan();

    // Public interface
    virtual void execute(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

private:
    // Private interface
    virtual void execute_cpu(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    virtual void execute_cpu_segmented(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    // Helper methods
    void generate_spin_frequency_table(
        dedisp_size nfreq,
        dedisp_size nsamp,
        dedisp_float dt);

    void fft_r2c(
        unsigned int n,
        unsigned int batch,
        size_t in_stride,
        size_t out_stride,
        float *in,
        float *out);

    void fft_r2c_inplace(
        unsigned int n,
        unsigned int batch,
        size_t stride,
        float *data);

    void fft_c2r(
        unsigned int n,
        unsigned int batch,
        size_t in_stride,
        size_t out_stride,
        float *in,
        float *out);

    void fft_c2r_inplace(
        unsigned int n,
        unsigned int batch,
        size_t stride,
        float *data);

    // Host arrays
    std::vector<dedisp_float> h_spin_frequencies; // size = nfreq
};

} // end namespace dedisp

#endif // H_FDD_CPU_PLAN_INCLUDE_GUARD