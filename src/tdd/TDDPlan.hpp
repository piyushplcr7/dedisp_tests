/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* Time Domain Dedispersion (TDD)
* is an optimized version of the original dedisp implementation.
*/
#include "GPUPlan.hpp"
#include "dedisperse/TDDKernel.hpp"

namespace dedisp
{

class TDDPlan : public GPUPlan {

public:
    // Constructor
    TDDPlan(
        size_type  nchans,
        float_type dt,
        float_type f0,
        float_type df,
        int device_idx = 0);

    // Destructor
    ~TDDPlan();

    // Public interface
    void set_gulp_size(size_type gulp_size);
    dedisp_size get_gulp_size() const { return m_gulp_size; };

    virtual void execute(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits,
        unsigned         flags = 0) override;

    void execute_adv(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        size_type        in_stride,
        byte_type*       out,
        size_type        out_nbits,
        size_type        out_stride);

    void execute_guru(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        size_type        in_stride,
        byte_type*       out,
        size_type        out_nbits,
        size_type        out_stride,
        dedisp_size      first_dm_idx,
        dedisp_size      dm_count);

private:
    // Size parameters
    dedisp_size  m_gulp_size;

    // DedispKernel
    DedispKernel m_kernel;

    dedisp_size compute_gulp_size();

    dedisp_size compute_max_nchans();

    void initialize_kernel();
};

} // end namespace dedisp