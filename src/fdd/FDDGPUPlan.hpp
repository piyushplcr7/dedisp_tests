// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef H_FDD_GPU_PLAN_INCLUDE_GUARD
#define H_FDD_GPU_PLAN_INCLUDE_GUARD

#include "GPUPlan.hpp"
#include "FDDCPUPlan.hpp"
#include "gpu_xccl.hpp"
#include "gpu_fft.hpp"
#include <pthread.h>
#include <sched.h>

// #define USECUFILE

#ifdef USECUFILE
#include <cufile.h>
#endif

extern size_t nsamp_fft;
extern size_t nsamps_computed;
extern size_t nsamp_padded;
extern int numGPUsLocal;
extern int total_cores;

#ifdef CUFILE
// Check cuFile API calls that return CUfileError_t
#define CK_CUFILE(call)                                                        \
do {                                                                           \
    CUfileError_t _st = (call);                                                \
    if (_st.err != CU_FILE_SUCCESS) {                                          \
        fprintf(stderr,                                                        \
                "cuFile error %s:%d: err=%d (%s)\n",                           \
                __FILE__, __LINE__, (int)_st.err, CUFILE_ERRSTR(_st.err));     \
        std::exit(EXIT_FAILURE);                                               \
    }                                                                          \
} while (0)
#endif

void pin_thread_to_core(int core_id) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
}


static inline size_t round_up_4k(size_t x) {
    const size_t A = 4096;
    return (x + (A - 1)) & ~(A - 1);
}

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
        int device_idx,
        int global_gpu_id_,
        int start_chan_,
        int end_chan_,
        int nchan_gpu_, 
        int start_chan_node_, 
        ncclComm_t comm_);

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

    void gpuDirectWrite(std::string path_string, void* buf, size_t size);

    void allocateMem(size_t nsamps);

    void determineBufferSizes(size_t nsamps);

    void setNdmBuffers(unsigned int ndm_bufs) {ndm_buffers = ndm_bufs;}

    unsigned int getNdmBuffers() {return ndm_buffers;}

    // Assumes same height of the source and destination memory layout in 2D
    void memcpy2D_internal(
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height);

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

    cu::DeviceMemory d_data_x_nu;

    int global_gpu_id = 0;
    int local_gpu_id = 0;
    int start_chan = 0;
    int end_chan = 0;
    int nchan_gpu = 0;
    int start_chan_node = 0;
    ncclComm_t comm;

    unsigned int nchan_fft_batch;
    unsigned int ndm_fft_batch;
    unsigned int nchan;
    unsigned int nchan_batch_max;
    unsigned int nchan_buffers;
    unsigned int ndm;
    unsigned int ndm_batch_max;
    unsigned int ndm_buffers;
    dedisp_size nchan_words_gulp;
    dedisp_size nchan_words;
    dedisp_size chans_per_word;
    size_t nsamp;
    size_t nfreq;
    size_t sizeof_data_t_nu;
    size_t sizeof_data_x_nu;
    size_t sizeof_data_x_dm;

    gpufftHandle plan_r2c;
    gpufftHandle plan_c2r;
};

} // end namespace dedisp

#endif // H_FDD_GPU_PLAN_INCLUDE_GUARD