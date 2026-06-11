// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef H_FDD_CPU_PLAN_INCLUDE_GUARD
#define H_FDD_CPU_PLAN_INCLUDE_GUARD

#include <vector>
#include <memory>

#include "Plan.hpp"

// Forward declarations (defined in dataHandlers/); only referenced by pointer/
// reference here, so the FITS/MPI headers are pulled in by FDDCPUPlan.cpp only.
class dataLoader;
class dataFile;

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

    // dataLoader-aware constructor (mirrors FDDGPUPlan) for the MPI + FITS driver
    FDDCPUPlan(const dataLoader& container, int device_index);

    // Equispaced DM list (host-only port of GPUPlan::generate_dm_list_equispaced)
    void generate_dm_list_equispaced(
        float_type dm_start,
        float_type dm_step,
        size_type  numdms);

    // Destructor
    ~FDDCPUPlan();

    // Public interface
    virtual void execute(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits,
        unsigned         flags = 0) override;

    // Output configuration + writers (mirror FDDGPUPlan; host-only file I/O)
    void setOutputParams(
        bool cleanout,
        bool fftout,
        bool multout,
        int out_nbits,
        char* outfile,
        int w,
        bool barycenter);

    void writeOutput(char* outfile, int w, bool barycenter, const std::vector<int>& inForOut);

    void writeInfs(char* outfile, const dataFile* file, size_t nsamps, double dt, int w,
                   bool barycenter = false, double blotoa = 0.0, double avgvoverc = 0.0);

    std::unique_ptr<float[]> output_buffer_;

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

    // Output parameters (mirror FDDGPUPlan), set by setOutputParams()
    bool   cleanout_   = false;
    bool   fftout_     = false;
    bool   multout_    = false;
    int    w_          = 0;
    bool   barycenter_ = false;
    char*  outfile_    = nullptr;

    size_t nsamps_          = 0; // total local time samples (from dataLoader ctor)
    size_t nsamp_fft_       = 0;
    size_t nsamps_computed_ = 0;
    size_t nsamp_padded_    = 0;
    double dt_              = 0.0;

    // Set by the dataLoader-aware constructor
    const dataLoader* container_ = nullptr;
};

} // end namespace dedisp

#endif // H_FDD_CPU_PLAN_INCLUDE_GUARD