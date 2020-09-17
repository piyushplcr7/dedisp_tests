#include "FDDGPUPlan.hpp"

#include <complex>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>

#include <assert.h>
#include <omp.h>
#include <cufft.h>

#include "common/dedisp_strings.h"
#include "unpack/unpack.h"
#include "dedisperse/FDDKernel.hpp"
#include "external/Stopwatch.h"

#include "common/helper.h"
#include "helper.h"
#include "chunk.h"

namespace dedisp
{

// Constructor
FDDGPUPlan::FDDGPUPlan(
    size_type  nchans,
    float_type dt,
    float_type f0,
    float_type df) :
    GPUPlan(nchans, dt, f0, df)
{
}

// Destructor
FDDGPUPlan::~FDDGPUPlan()
{}

// Public interface
void FDDGPUPlan::execute(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits)
{
    char* use_segmented_str = getenv("USE_SEGMENTED");
    bool use_segmented = !use_segmented_str ? false : atoi(use_segmented_str);
    if (use_segmented)
    {
        std::cout << ">> Running segmented GPU implementation" << std::endl;
        execute_gpu_segmented(nsamps, in, in_nbits, out, out_nbits);
    } else {
        std::cout << ">> Running GPU implementation" << std::endl;
        execute_gpu(nsamps, in, in_nbits, out, out_nbits);
    }
}

// Private interface
void FDDGPUPlan::execute_gpu(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits)
{
    enum {
        BITS_PER_BYTE  = 8,
        BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
    };

    assert(in_nbits == 8);
    assert(out_nbits == 32);

    // Parameters
    float dt           = m_dt;          // sample time
    unsigned int nchan = m_nchans;      // number of observering frequencies
    unsigned int nsamp = nsamps;        // number of time samples
    unsigned int nfreq = (nsamp/2 + 1); // number of spin frequencies
    unsigned int ndm   = m_dm_count;    // number of DMs

    // Compute the number of output samples
    unsigned int nsamp_computed = nsamp - m_max_delay;

    // Use zero-padded FFT
    bool use_zero_padding = true;

    // Compute padded number of samples (for r2c transformation)
    unsigned int nsamp_fft    = use_zero_padding
                                ? round_up(nsamp + 1, 16384)
                                : nsamp;
    unsigned int nsamp_padded = round_up(nsamp_fft + 1, 1024);
    std::cout << debug_str << std::endl;
    std::cout << "nsamp_fft    = " << nsamp_fft << std::endl;
    std::cout << "nsamp_padded = " << nsamp_padded << std::endl;

    // Maximum number of DMs computed in one gulp
    unsigned int ndm_batch_max = 256;
    unsigned int ndm_fft_batch = 32;
                 ndm_fft_batch = std::min(ndm_batch_max, ndm_fft_batch);
    unsigned int ndm_buffers   = 1;
                 ndm_buffers   = std::min(ndm_buffers, (unsigned int) ((ndm + ndm_batch_max) / ndm_batch_max));

    // Maximum number of channels processed in one gulp
    unsigned int nchan_batch_max = 128;
    unsigned int nchan_fft_batch = 64;
    unsigned int nchan_buffers   = 2;

    // Verbose iteration reporting
    bool enable_verbose_iteration_reporting = false;

    // Compute derived counts
    dedisp_size out_bytes_per_sample = out_nbits / (sizeof(dedisp_byte) *
                                                    BITS_PER_BYTE);
    dedisp_size chans_per_word = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;

    // The number of channel words in the input
    dedisp_size nchan_words = nchan / chans_per_word;

    // The number of channel words proccessed in one gulp
    dedisp_size nchan_words_gulp = nchan_batch_max / chans_per_word;

    // Events, markers, timers
    cu::Event eStartGPU, eEndGPU;
    cu::Marker mAllocMem("Allocate host and device memory", cu::Marker::black);
    cu::Marker mCopyMem("Copy CUDA mem to CPU mem", cu::Marker::black);
    cu::Marker mPrepFFT("cufft Plan Many", cu::Marker::yellow);
    cu::Marker mPrepSpinf("spin Frequency generation", cu::Marker::blue);
    cu::Marker mDelayTable("Delay table copy", cu::Marker::black);
    cu::Marker mExeGPU("Dedisp fdd execution on GPU", cu::Marker::green);
    std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> input_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();

    // Prepare cuFFT plans
    std::cout << fft_plan_str << std::endl;
    mPrepFFT.start();
    cufftHandle plan_r2c, plan_c2r;
    int n[] = {(int) nsamp_fft};
    int rnembed[] = {(int) nsamp_padded};   // width in real elements
    int cnembed[] = {(int) nsamp_padded/2}; // width in complex elements
    std::thread thread_r2c = std::thread([&]()
    {
        cufftResult result = cufftPlanMany(
            &plan_r2c,              // plan
            1, n,                   // rank, n
            rnembed, 1, rnembed[0], // inembed, istride, idist
            cnembed, 1, cnembed[0], // onembed, ostride, odist
            CUFFT_R2C,              // type
            nchan_fft_batch);       // batch
        if (result != CUFFT_SUCCESS)
        {
            throw std::runtime_error("Error creating real to complex FFT plan.");
        }
        cufftSetStream(plan_r2c, *executestream);
    });
    std::thread thread_c2r = std::thread([&]()
    {
        cufftResult result = cufftPlanMany(
            &plan_c2r,              // plan
            1, n,                   // rank, n
            cnembed, 1, cnembed[0], // inembed, istride, idist
            rnembed, 1, rnembed[0], // onembed, ostride, odist
            CUFFT_C2R,              // type
            ndm_fft_batch);         // batch
        if (result != CUFFT_SUCCESS)
        {
            throw std::runtime_error("Error creating complex to real FFT plan.");
        }
        cufftSetStream(plan_c2r, *executestream);
    });

    // Wait for cuFFT plans to be created
    if (thread_r2c.joinable()) { thread_r2c.join(); }
    if (thread_c2r.joinable()) { thread_c2r.join(); }
    mPrepFFT.end();

    // Determine the amount of memory to use
    size_t memory_total = m_device->get_total_memory();
    size_t memory_free = m_device->get_free_memory();
    size_t sizeof_data_t_nu = 1ULL * nsamp * nchan_words_gulp * sizeof(dedisp_word);
    size_t sizeof_data_t_dm = 1ULL * ndm * nsamp_padded * sizeof(float);
    size_t sizeof_data_x_nu = 1ULL * nchan_batch_max * nsamp_padded * sizeof(float);
    size_t sizeof_data_x_dm = 1ULL * ndm_batch_max * nsamp_padded * sizeof(float);
    size_t memory_required  = sizeof_data_t_nu * nchan_buffers +
                              sizeof_data_x_nu * 1 +
                              sizeof_data_x_dm * ndm_buffers;
    size_t memory_reserved  = 0.05 * memory_total;

    while ((ndm_buffers * ndm_batch_max) < ndm &&
           (memory_required + memory_reserved + sizeof_data_x_dm) < memory_free)
    {
        ndm_buffers++;
        memory_required = sizeof_data_t_nu * nchan_buffers +
                          sizeof_data_x_nu * 1 +
                          sizeof_data_x_dm * (ndm_buffers);
    };

    // Debug
    std::cout << debug_str << std::endl;
    std::cout << "ndm_buffers     = " << ndm_buffers << " x " << ndm_batch_max << " DMs" << std::endl;
    std::cout << "nchan_buffers   = " << nchan_buffers << " x " << nchan_batch_max << " channels" << std::endl;
    std::cout << "Memory total    = " << memory_total / std::pow(1024, 3) << " Gb" << std::endl;
    std::cout << "Memory free     = " << memory_free  / std::pow(1024, 3) << " Gb" << std::endl;
    std::cout << "Memory required = " << memory_required / std::pow(1024, 3) << " Gb" << std::endl;

    // Allocate memory
    std::cout << memory_alloc_str << std::endl;
    mAllocMem.start();
    /*
        The buffers are used as follows:
        1) copy into page-locked buffer: in -> memcpyHtoH -> h_data_t_nu
        2) copy to device: h_data_t_nu -> memcopyHtoD -> d_data_t_nu
        3) unpack and transpose: d_data_t_nu -> transpose_unpack -> d_data_tf_nu
        4) in-place Fourier transform: d_data_x_nu -> fft_r2c -> d_data_x_nu
        5) apply dedispersion: d_data_x_nu -> dedispserse -> d_data_x_dm
        6) in-place Fourier transform: d_data_x_dm -> fft_c2r -> d_data_x_dm
        7) copy to host: d_data_x_dm -> memcpyDtoH -> h_data_t_dm

        The suffixes have the following meaning:
        * The _t indicates that the buffer contains time domain data
        * The _f indicates that the buffer contains Fourier domain data
        * The _x indicates that the type of data various throughout processing
        * The _nu indicates input data with observing frequencies as outer dimension
        * The _dm indicates output data with DM as outer dimension

        The vectors (with _ suffix) are used to implement multiple-buffering
    */
    std::vector<cu::HostMemory>   h_data_t_nu_(nchan_buffers);
    std::vector<cu::DeviceMemory> d_data_t_nu_(nchan_buffers);
                cu::DeviceMemory  d_data_x_nu(sizeof_data_x_nu);
    std::vector<cu::DeviceMemory> d_data_x_dm_(ndm_buffers);
                cu::HostMemory    h_data_t_dm(sizeof_data_t_dm);
    for (unsigned int i = 0; i < nchan_buffers; i++)
    {
        h_data_t_nu_[i].resize(sizeof_data_t_nu);
        d_data_t_nu_[i].resize(sizeof_data_t_nu);
    }
    for (unsigned int i = 0; i < ndm_buffers; i++)
    {
        d_data_x_dm_[i].resize(sizeof_data_x_dm);
    }
    mAllocMem.end();

    // Generate spin frequency table
    mPrepSpinf.start();
    if (h_spin_frequencies.size() != nfreq)
    {
        generate_spin_frequency_table(nfreq, nsamp, dt);
    }
    mPrepSpinf.end();

    // Initialize FDDKernel
    FDDKernel kernel;
    mDelayTable.start();
    kernel.copy_delay_table(
        d_delay_table,
        m_nchans * sizeof(dedisp_float),
        0, *htodstream);
    mDelayTable.end();
    init_timer->Pause();

    struct ChannelData
    {
        unsigned int ichan_start;
        unsigned int ichan_end;
        unsigned int nchan_current;
        void* h_in_ptr;
        void* d_in_ptr;
        cu::Event inputStart, inputEnd;
        cu::Event preprocessingStart, preprocessingEnd;
        cu::Event outputStart, outputEnd;
    };

    unsigned int nchan_jobs = (nchan + nchan_batch_max) / nchan_batch_max;
    std::vector<ChannelData> channel_jobs(nchan_jobs);

    for (unsigned job_id = 0; job_id < nchan_jobs; job_id++)
    {
        ChannelData& job = channel_jobs[job_id];
        job.ichan_start   = job_id == 0 ? 0 : channel_jobs[job_id - 1].ichan_end;
        job.nchan_current = std::min(nchan_batch_max, nchan - job.ichan_start);
        job.ichan_end     = job.ichan_start + job.nchan_current;
        job.h_in_ptr      = h_data_t_nu_[job_id % nchan_buffers];
        job.d_in_ptr      = d_data_t_nu_[job_id % nchan_buffers];
        if (job.nchan_current == 0) {
            channel_jobs.pop_back();
        }
    }

    struct DMData{
        unsigned int idm_start;
        unsigned int idm_end;
        unsigned int ndm_current;
        float* h_out_ptr;
        cu::DeviceMemory* d_data_x_dm;
        cu::Event inputStart, inputEnd;
        cu::Event dedispersionStart, dedispersionEnd;
        cu::Event postprocessingStart, postprocessingEnd;
        cu::Event outputStart, outputEnd;
    };

    unsigned int ndm_jobs = (ndm + ndm_batch_max) / ndm_batch_max;
    std::vector<DMData> dm_jobs(ndm_jobs);

    for (unsigned job_id = 0; job_id < ndm_jobs; job_id++)
    {
        DMData& job = dm_jobs[job_id];
        job.idm_start   = job_id == 0 ? 0 : dm_jobs[job_id - 1].idm_end;
        job.ndm_current = std::min(ndm_batch_max, ndm - job.idm_start);
        job.idm_end     = job.idm_start + job.ndm_current;
        job.d_data_x_dm = &d_data_x_dm_[job_id % ndm_buffers];
        if (job.ndm_current == 0)
        {
            dm_jobs.pop_back();
        }
    }

    std::cout << fdd_dedispersion_str << std::endl;
    htodstream->record(eStartGPU);
    mExeGPU.start();

    // Process all dm batches
    for (unsigned dm_job_id_outer = 0; dm_job_id_outer < dm_jobs.size(); dm_job_id_outer += ndm_buffers)
    {
        // Process all channel batches
        for (unsigned channel_job_id = 0; channel_job_id < channel_jobs.size(); channel_job_id++)
        {
            auto& channel_job = channel_jobs[channel_job_id];

            // Info
            if (enable_verbose_iteration_reporting)
            {
                std::cout << "Processing channel " << channel_job.ichan_start << " to " << channel_job.ichan_end << std::endl;
            }

            // Channel input size
            dedisp_size dst_stride = nchan_words_gulp * sizeof(dedisp_word);
            dedisp_size src_stride = nchan_words * sizeof(dedisp_word);

            // Copy the input data for the first job
            if (channel_job_id == 0)
            {
                dedisp_size gulp_chan_byte_idx = (channel_job.ichan_start/chans_per_word) * sizeof(dedisp_word);
                memcpy2D(
                    channel_job.h_in_ptr,    // dst
                    dst_stride,              // dst width
                    in + gulp_chan_byte_idx, // src
                    src_stride,              // src width
                    dst_stride,              // width bytes
                    nsamp);                  // height
                htodstream->record(channel_job.inputStart);
                htodstream->memcpyHtoDAsync(
                    channel_job.d_in_ptr, // dst
                    channel_job.h_in_ptr, // src
                    nsamp * dst_stride);  // size
                htodstream->record(channel_job.inputEnd);
            }
            executestream->waitEvent(channel_job.inputEnd);

            // Transpose and upack the data
            executestream->record(channel_job.preprocessingStart);
            transpose_unpack(
                (dedisp_word*) channel_job.d_in_ptr, // d_in
                nchan_words_gulp,                    // input width
                nsamp,                               // input height
                nchan_words_gulp,                    // in_stride
                nsamp_padded,                        // out_stride
                d_data_x_nu,                         // d_out
                in_nbits, 32,                        // in_nbits, out_nbits
                1.0/nchan,                           // scale
                *executestream);                     // stream

            // Apply zero padding
            auto dst_ptr = ((float *) d_data_x_nu.data()) + nsamp;
            unsigned int nsamp_padding = nsamp_padded - nsamp;
            cu::checkError(cudaMemset2DAsync(
                dst_ptr,                       // devPtr
                nsamp_padded * sizeof(float),  // pitch
                0,                             // value
                nsamp_padding * sizeof(float), // width
                nchan_batch_max,               // height
                *executestream
            ));

            // FFT data (real to complex) along time axis
            for (unsigned int i = 0; i < nchan_batch_max/nchan_fft_batch; i++)
            {
                cufftReal    *idata = (cufftReal *) d_data_x_nu.data() + i * nsamp_padded * nchan_fft_batch;
                cufftComplex *odata = (cufftComplex *) idata;
                cufftExecR2C(plan_r2c, idata, odata);
            }
            executestream->record(channel_job.preprocessingEnd);

            // Process DM batches
            for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
            {
                unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
                if (dm_job_id >= dm_jobs.size())
                {
                    break;
                }
                auto& dm_job = dm_jobs[dm_job_id];

                // Info
                if (enable_verbose_iteration_reporting)
                {
                    std::cout << "Processing DM " << dm_job.idm_start << " to " << dm_job.idm_end << std::endl;
                }

                // Initialize output to zero
                if (channel_job_id == 0)
                {
                    // Wait for previous output copy to finish
                    if (dm_job_id_outer > 0)
                    {
                        dm_job.outputEnd.synchronize();
                    }

                    dm_job.d_data_x_dm->zero(*executestream);
                }

                // Wait for temporary output from previous job to be copied
                if (channel_job_id > (nchan_buffers-1))
                {
                    auto& job_previous = channel_jobs[channel_job_id - nchan_buffers];
                    job_previous.outputEnd.synchronize();
                }

                // Dedispersion in frequency domain
                executestream->record(dm_job.dedispersionStart);
                auto d_out = (dedisp_float2 *) dm_job.d_data_x_dm->data();
                kernel.launch(
                    dm_job.ndm_current,        // ndm
                    nfreq,                     // nfreq
                    channel_job.nchan_current, // nchan
                    dt,                        // dt
                    d_spin_frequencies,        // d_spin_frequencies
                    d_dm_list,                 // d_dm_list
                    d_data_x_nu,               // d_in
                    d_out,                     // d_out
                    nsamp_padded/2,            // in stride
                    nsamp_padded/2,            // out stride
                    dm_job.idm_start,          // idm_start
                    dm_job.idm_end,            // idm_end
                    channel_job.ichan_start,   // ichan_start
                    *executestream);           // stream
                executestream->record(dm_job.dedispersionEnd);
            } // end for dm_job_id_inner

            // Copy the input data for the next job (if any)
            unsigned channel_job_id_next = channel_job_id + 1;
            if (channel_job_id_next < channel_jobs.size())
            {
                auto& channel_job_next = channel_jobs[channel_job_id_next];
                dedisp_size gulp_chan_byte_idx = (channel_job_next.ichan_start/chans_per_word) * sizeof(dedisp_word);
                memcpy2D(
                    channel_job_next.h_in_ptr,  // dst
                    dst_stride,                 // dst width
                    in + gulp_chan_byte_idx,    // src
                    src_stride,                 // src width
                    dst_stride,                 // width bytes
                    nsamp);                     // height
                htodstream->record(channel_job_next.inputStart);
                htodstream->memcpyHtoDAsync(
                    channel_job_next.d_in_ptr, // dst
                    channel_job_next.h_in_ptr, // src
                    nsamp * dst_stride);       // size
                htodstream->record(channel_job_next.inputEnd);
            }

            // Wait for current batch to finish
            executestream->synchronize();

            // Add input and preprocessing time for the current channel job
            input_timer->Add(channel_job.inputEnd.elapsedTime(channel_job.inputStart));
            preprocessing_timer->Add(channel_job.preprocessingEnd.elapsedTime(channel_job.preprocessingStart));

            // Add dedispersion time for current dm jobs
            for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
            {
                unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
                if (dm_job_id >= dm_jobs.size())
                {
                    break;
                }
                auto& dm_job = dm_jobs[dm_job_id];
                dedispersion_timer->Add(dm_job.dedispersionEnd.elapsedTime(dm_job.dedispersionStart));
            }
        } // end for ichan_start

        // Output DM batches
        for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
        {
            unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
            if (dm_job_id >= dm_jobs.size())
            {
                break;
            }
            auto& dm_job = dm_jobs[dm_job_id];

            // Get pointer to DM output data on host and on device
            dedisp_size dm_stride = 1ULL * nsamp_padded * out_bytes_per_sample;
            dedisp_size dm_offset = 1ULL * dm_job.idm_start * dm_stride;
            auto* h_out = (void *) (((size_t) h_data_t_dm.data()) + dm_offset);
            auto *d_out = (float *) dm_job.d_data_x_dm->data();

            // Fourier transform results back to time domain
            executestream->record(dm_job.postprocessingStart);
            for (unsigned int i = 0; i < ndm_batch_max/ndm_fft_batch; i++)
            {
                cufftReal    *odata = (cufftReal *) d_out + i * nsamp_padded * ndm_fft_batch;
                cufftComplex *idata = (cufftComplex *) odata;
                cufftExecC2R(plan_c2r, idata, odata);
            }

            // FFT scaling
            kernel.scale(
                dm_job.ndm_current, // height
                nsamp_padded,       // width
                nsamp_padded,       // stride
                1.0f / nsamp_fft,   // scale
                d_out,              // d_data
                *executestream);    // stream
            executestream->record(dm_job.postprocessingEnd);

            // Copy output
            dtohstream->waitEvent(dm_job.postprocessingEnd);
            dtohstream->record(dm_job.outputStart);
            dedisp_size size = 1ULL * dm_job.ndm_current * dm_stride;
            dtohstream->memcpyDtoHAsync(
                h_out, // dst
                d_out, // src
                size); // size
            dtohstream->record(dm_job.outputEnd);
        } // end for dm_job_id_inner
    } // end for dm_job_id_outer

    // Wait for final memory transfer
    dtohstream->record(eEndGPU);
    mExeGPU.end(eEndGPU);
    dtohstream->synchronize();

    // Copy output
    std::cout << copy_output_str << std::endl;
    mCopyMem.start();
    output_timer->Start();
    dedisp_size dst_stride = 1ULL * nsamp_computed * out_bytes_per_sample;
    dedisp_size src_stride = 1ULL * nsamp_padded * out_bytes_per_sample;
    memcpy2D(
        out,         // dst
        dst_stride,  // dst width
        h_data_t_dm, // src
        src_stride,  // src width
        dst_stride,  // width bytes
        ndm);        // height
    output_timer->Pause();
    mCopyMem.end();
    total_timer->Pause();

    // Accumulate postprocessing time for all dm jobs
    for (auto& job : dm_jobs)
    {
        postprocessing_timer->Add(job.postprocessingEnd.elapsedTime(job.postprocessingStart));
    }

    // Print timings
    long double runtime_time = preprocessing_timer->Milliseconds() + dedispersion_timer->Milliseconds() + postprocessing_timer->Milliseconds();
    runtime_time *= 1e-3; //seconds
    std::stringstream runtime_time_string;
    runtime_time_string << std::fixed;
    runtime_time_string << runtime_time;

    std::cout << timings_str << std::endl;
    std::cout << init_time_str           << init_timer->ToString() << " sec." << std::endl;
    std::cout << input_memcpy_time_str   << input_timer->ToString() << " sec." << std::endl;
    std::cout << preprocessing_time_str  << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << dedispersion_time_str   << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << postprocessing_time_str << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << output_memcpy_time_str  << output_timer->ToString() << " sec." << std::endl;
    std::cout << runtime_time_str        << runtime_time_string.str() << " sec." << std::endl;
    std::cout << total_time_str          << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}

void FDDGPUPlan::execute_gpu_segmented(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits)
{
    enum {
        BITS_PER_BYTE  = 8,
        BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
    };

    assert(in_nbits == 8);
    assert(out_nbits == 32);

    // Parameters
    float dt           = m_dt;          // sample time
    unsigned int nchan = m_nchans;      // number of observering frequencies
    unsigned int nsamp = nsamps;        // number of time samples
    unsigned int nfreq = (nsamp/2 + 1); // number of spin frequencies
    unsigned int ndm   = m_dm_count;    // number of DMs
    unsigned int nfft  = 16384;         // number of samples processed in a segment

    // Compute the number of output samples
    unsigned int nsamp_computed = nsamp - m_max_delay;

    // Compute the number of chunks
    unsigned int nsamp_dm   = std::ceil(m_max_delay);
    float min_efficiency    = 0.8;
    while ((nfft * (1.0 - min_efficiency)) < nsamp_dm) { nfft *= 2; };
    unsigned int nsamp_good = nfft - nsamp_dm;
    unsigned int nchunk     = std::ceil((float) nsamp / nsamp_good);

    // For every channel, a buffer of nsamp_padded scalar elements long is used,
    // resulting in a two-dimensional buffers of size buffer[nchan][nsamp_padded]
    // Every row of is divided into chunks of nfreq_chunk_padded complex elements,
    // thus the implicit dimensions are buffer[nchan][nchunk][nfreq_chunk_padded],
    // of which only nfreq_chunk elements in the innermost dimension are used.
    unsigned int nfreq_chunk        = std::ceil(nfft / 2) + 1;
    unsigned int nfreq_chunk_padded = round_up(nfreq_chunk + 1, 1024);
    unsigned int nsamp_padded       = nchunk * (nfreq_chunk_padded * 2);

    // Debug
    std::cout << debug_str << std::endl;
    std::cout << "nfft               = " << nfft << std::endl;
    std::cout << "nsamp_dm           = " << nsamp_dm << std::endl;
    std::cout << "nsamp_good         = " << nsamp_good << std::endl;
    std::cout << "nchunk             = " << nchunk << std::endl;
    std::cout << "nfreq_chunk        = " << nfreq_chunk << std::endl;
    std::cout << "nfreq_chunk_padded = " << nfreq_chunk_padded << std::endl;
    std::cout << "nsamp_padded       = " << nsamp_padded << std::endl;

    // Maximum number of DMs computed in one gulp
    unsigned int ndm_batch_max = 32;
    unsigned int ndm_buffers   = 8;
                 ndm_buffers   = std::min(ndm_buffers, (unsigned int) ((ndm + ndm_batch_max) / ndm_batch_max));

    // Maximum number of channels processed in one gulp
    unsigned int nchan_batch_max = 32;
    unsigned int nchan_buffers   = 2;

    // Verbose iteration reporting
    bool enable_verbose_iteration_reporting = false;

    // Compute derived counts
    dedisp_size out_bytes_per_sample = out_nbits / (sizeof(dedisp_byte) *
                                                    BITS_PER_BYTE);
    dedisp_size chans_per_word = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;

    // The number of channel words in the input
    dedisp_size nchan_words = nchan / chans_per_word;

    // The number of channel words proccessed in one gulp
    dedisp_size nchan_words_gulp = nchan_batch_max / chans_per_word;

    // Events, markers, timers
    cu::Event eStartGPU, eEndGPU;
    cu::Marker mAllocMem("Allocate host and device memory", cu::Marker::black);
    cu::Marker mCopyMem("Copy CUDA mem to CPU mem", cu::Marker::black);
    cu::Marker mPrepFFT("cufft Plan Many", cu::Marker::yellow);
    cu::Marker mPrepSpinf("spin Frequency generation", cu::Marker::blue);
    cu::Marker mDelayTable("Delay table copy", cu::Marker::black);
    cu::Marker mExeGPU("Dedisp fdd execution on GPU", cu::Marker::green);
    std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> input_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();

    // Allocate memory
    std::cout << memory_alloc_str << std::endl;
    mAllocMem.start();
    cu::HostMemory   h_data_t_dm(ndm * nsamp_padded * sizeof(float));
    cu::DeviceMemory d_data_t_nu(nchan_batch_max * nsamp_padded * sizeof(float));
    cu::DeviceMemory d_data_f_nu(nchan_batch_max * nsamp_padded * sizeof(float));
    std::vector<cu::HostMemory>   h_data_t_nu_(nchan_buffers);
    std::vector<cu::DeviceMemory> d_data_t_nu_(nchan_buffers);
    std::vector<cu::DeviceMemory> d_data_f_dm_(ndm_buffers);
    std::vector<cu::DeviceMemory> d_data_t_dm_(ndm_buffers);
    for (unsigned int i = 0; i < nchan_buffers; i++)
    {
        h_data_t_nu_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
        d_data_t_nu_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
    }
    for (unsigned int i = 0; i < ndm_buffers; i++)
    {
        d_data_f_dm_[i].resize(ndm_batch_max * nsamp_padded * sizeof(float));
        d_data_t_dm_[i].resize(ndm_batch_max * nsamp_padded * sizeof(float));
    }
    mAllocMem.end();

    // Prepare cuFFT plans
    std::cout << fft_plan_str << std::endl;
    mPrepFFT.start();
    cufftHandle plan_r2c, plan_c2r;
    int n[] = {(int) nfft};
    std::thread thread_r2c = std::thread([&]()
    {
        int inembed[] = {(int) nsamp_good};
        int onembed[] = {(int) nfreq_chunk_padded};
        cufftResult result = cufftPlanMany(
            &plan_r2c,              // plan
            1, n,                   // rank, n
            inembed, 1, inembed[0], // inembed, istride, idist
            onembed, 1, onembed[0], // onembed, ostride, odist
            CUFFT_R2C,              // type
            nchunk);                // batch
        if (result != CUFFT_SUCCESS)
        {
            throw std::runtime_error("Error creating real to complex FFT plan.");
        }
        cufftSetStream(plan_r2c, *executestream);
    });
    std::thread thread_c2r = std::thread([&]()
    {
        int inembed[] = {(int) nfreq_chunk_padded};
        int onembed[] = {(int) nfreq_chunk_padded*2};
        cufftResult result = cufftPlanMany(
            &plan_c2r,              // plan
            1, n,                   // rank, n
            inembed, 1, inembed[0], // inembed, istride, idist
            onembed, 1, onembed[0], // onembed, ostride, odist
            CUFFT_C2R,              // type
            nchunk);                // batch
        if (result != CUFFT_SUCCESS)
        {
            throw std::runtime_error("Error creating complex to real FFT plan.");
        }
        cufftSetStream(plan_c2r, *executestream);
    });

    // Compute chunks
    std::vector<Chunk> chunks(nchunk);
    unsigned int nfreq_computed;
    compute_chunks(
        nsamp, nsamp_good, nfft,
        nfreq_chunk_padded, nfreq_computed, chunks);

    // Wait for cuFFT plans to be created
    if (thread_r2c.joinable()) { thread_r2c.join(); }
    if (thread_c2r.joinable()) { thread_c2r.join(); }
    mPrepFFT.end();

    // Generate spin frequency table
    mPrepSpinf.start();
    if (h_spin_frequencies.size() != nsamp_padded)
    {
        // Generate spin frequencies on the host
        h_spin_frequencies.resize(nsamp_padded);
        generate_spin_frequency_table_chunks(
            chunks, h_spin_frequencies,
            nfreq_chunk, nfreq_chunk_padded,
            nfft, dt);

        // Copy segmented spin frequencies to the GPU
        d_spin_frequencies.resize(h_spin_frequencies.size() * sizeof(float));
        htodstream->memcpyHtoDAsync(d_spin_frequencies, h_spin_frequencies.data(), d_spin_frequencies.size());
    }
    mPrepSpinf.end();

    // Initialize FDDKernel
    FDDKernel kernel;
    mDelayTable.start();
    kernel.copy_delay_table(
        d_delay_table,
        m_nchans * sizeof(dedisp_float),
        0, *htodstream);
    mDelayTable.end();
    init_timer->Pause();

    struct ChannelData
    {
        unsigned int ichan_start;
        unsigned int ichan_end;
        unsigned int nchan_current;
        void* h_in_ptr;
        void* d_in_ptr;
        cu::Event inputStart, inputEnd;
        cu::Event preprocessingStart, preprocessingEnd;
        cu::Event outputStart, outputEnd;
    };

    unsigned int nchan_jobs = (nchan + nchan_batch_max) / nchan_batch_max;
    std::vector<ChannelData> channel_jobs(nchan_jobs);

    for (unsigned job_id = 0; job_id < nchan_jobs; job_id++)
    {
        ChannelData& job = channel_jobs[job_id];
        job.ichan_start   = job_id == 0 ? 0 : channel_jobs[job_id - 1].ichan_end;
        job.nchan_current = std::min(nchan_batch_max, nchan - job.ichan_start);
        job.ichan_end     = job.ichan_start + job.nchan_current;
        job.h_in_ptr      = h_data_t_nu_[job_id % nchan_buffers];
        job.d_in_ptr      = d_data_t_nu_[job_id % nchan_buffers];
        if (job.nchan_current == 0) {
            channel_jobs.pop_back();
        }
    }

    struct DMData{
        unsigned int idm_start;
        unsigned int idm_end;
        unsigned int ndm_current;
        float* h_in_ptr;
        dedisp_float2* d_data_f_dm_ptr;
        dedisp_float2* d_data_t_dm_ptr;
        cu::Event inputStart, inputEnd;
        cu::Event dedispersionStart, dedispersionEnd;
        cu::Event postprocessingStart, postprocessingEnd;
        cu::Event outputStart, outputEnd;
    };

    unsigned int ndm_jobs = (ndm + ndm_batch_max) / ndm_batch_max;
    std::vector<DMData> dm_jobs(ndm_jobs);

    for (unsigned job_id = 0; job_id < ndm_jobs; job_id++)
    {
        DMData& job = dm_jobs[job_id];
        job.idm_start   = job_id == 0 ? 0 : dm_jobs[job_id - 1].idm_end;
        job.ndm_current = std::min(ndm_batch_max, ndm - job.idm_start);
        job.idm_end     = job.idm_start + job.ndm_current;
        job.d_data_f_dm_ptr   = d_data_f_dm_[job_id % ndm_buffers];
        job.d_data_t_dm_ptr   = d_data_t_dm_[job_id % ndm_buffers];
        if (job.ndm_current == 0)
        {
            dm_jobs.pop_back();
        }
    }

    std::cout << fdd_dedispersion_str << std::endl;
    htodstream->record(eStartGPU);
    mExeGPU.start();

    // Process all dm batches
    for (unsigned dm_job_id_outer = 0; dm_job_id_outer < dm_jobs.size(); dm_job_id_outer += ndm_buffers)
    {
        // Process all channel batches
        for (unsigned channel_job_id = 0; channel_job_id < channel_jobs.size(); channel_job_id++)
        {
            auto& channel_job = channel_jobs[channel_job_id];

            // Info
            if (enable_verbose_iteration_reporting)
            {
                std::cout << "Processing channel " << channel_job.ichan_start << " to " << channel_job.ichan_end << std::endl;
            }

            // Channel input size
            dedisp_size dst_stride = nchan_words_gulp * sizeof(dedisp_word);
            dedisp_size src_stride = nchan_words * sizeof(dedisp_word);

            // Copy the input data for the first job
            if (channel_job_id == 0)
            {
                dedisp_size gulp_chan_byte_idx = (channel_job.ichan_start/chans_per_word) * sizeof(dedisp_word);
                memcpy2D(
                    channel_job.h_in_ptr,    // dst
                    dst_stride,              // dst width
                    in + gulp_chan_byte_idx, // src
                    src_stride,              // src width
                    dst_stride,              // width bytes
                    nsamp);                  // height
                htodstream->record(channel_job.inputStart);
                htodstream->memcpyHtoDAsync(
                    channel_job.d_in_ptr, // dst
                    channel_job.h_in_ptr, // src
                    nsamp * dst_stride);  // size
                htodstream->record(channel_job.inputEnd);
            }
            executestream->waitEvent(channel_job.inputEnd);

            // Transpose and upack the data
            executestream->record(channel_job.preprocessingStart);
            transpose_unpack(
                (dedisp_word*) channel_job.d_in_ptr, // d_in
                nchan_words_gulp,                    // input width
                nsamp,                               // input height
                nchan_words_gulp,                    // in_stride
                nsamp_padded,                        // out_stride
                d_data_t_nu,                         // d_out
                in_nbits, 32,                        // in_nbits, out_nbits
                1.0/nchan,                           // scale
                *executestream);                     // stream

            // Apply zero padding
            auto dst_ptr = ((float *) d_data_t_nu.data()) + nsamp;
            unsigned int nsamp_padding = nsamp_padded - nsamp;
            cu::checkError(cudaMemset2DAsync(
                dst_ptr,                       // devPtr
                nsamp_padded * sizeof(float),  // pitch
                0,                             // value
                nsamp_padding * sizeof(float), // width
                nchan_batch_max,               // height
                *executestream
            ));

            // FFT data (real to complex) along time axis
            for (unsigned int ichan = 0; ichan < channel_job.nchan_current; ichan++)
            {
                auto *idata = (cufftReal *) d_data_t_nu.data() + (1ULL * ichan * nsamp_padded);
                auto *odata = (cufftComplex *) d_data_f_nu.data() + (1ULL * ichan * nsamp_padded/2);
                cufftExecR2C(plan_r2c, idata, odata);
            }
            executestream->record(channel_job.preprocessingEnd);

            // Initialize output to zero
            if (channel_job_id == 0)
            {
                // Wait for all previous output copies to finish
                dtohstream->synchronize();

                for (cu::DeviceMemory& d_data_out : d_data_f_dm_)
                {
                    // Use executestream to make sure dedispersion
                    // starts only after initializing the output buffer
                    d_data_out.zero(*executestream);
                }
            }

            // Process DM batches
            for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
            {
                unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
                if (dm_job_id >= dm_jobs.size())
                {
                    break;
                }
                auto& dm_job = dm_jobs[dm_job_id];

                // Info
                if (enable_verbose_iteration_reporting)
                {
                    std::cout << "Processing DM " << dm_job.idm_start << " to " << dm_job.idm_end << std::endl;
                }

                // Wait for temporary output from previous job to be copied
                if (channel_job_id > (nchan_buffers-1))
                {
                    auto& job_previous = channel_jobs[channel_job_id - nchan_buffers];
                    job_previous.outputEnd.synchronize();
                }

                // Dedispersion in frequency domain
                executestream->record(dm_job.dedispersionStart);
                kernel.launch(
                    dm_job.ndm_current,        // ndm
                    nfreq,                     // nfreq
                    channel_job.nchan_current, // nchan
                    dt,                        // dt
                    d_spin_frequencies,        // d_spin_frequencies
                    d_dm_list,                 // d_dm_list
                    d_data_f_nu,               // d_in
                    dm_job.d_data_f_dm_ptr,    // d_out
                    nsamp_padded/2,            // in stride
                    nsamp_padded/2,            // out stride
                    dm_job.idm_start,          // idm_start
                    dm_job.idm_end,            // idm_end
                    channel_job.ichan_start,   // ichan_start
                    *executestream);           // stream
                executestream->record(dm_job.dedispersionEnd);
            } // end for dm_job_id_inner

            // Copy the input data for the next job (if any)
            unsigned channel_job_id_next = channel_job_id + 1;
            if (channel_job_id_next < channel_jobs.size())
            {
                auto& channel_job_next = channel_jobs[channel_job_id_next];
                dedisp_size gulp_chan_byte_idx = (channel_job_next.ichan_start/chans_per_word) * sizeof(dedisp_word);
                memcpy2D(
                    channel_job_next.h_in_ptr,  // dst
                    dst_stride,                 // dst width
                    in + gulp_chan_byte_idx,    // src
                    src_stride,                 // src width
                    dst_stride,                 // width bytes
                    nsamp);                     // height
                htodstream->record(channel_job_next.inputStart);
                htodstream->memcpyHtoDAsync(
                    channel_job_next.d_in_ptr, // dst
                    channel_job_next.h_in_ptr, // src
                    nsamp * dst_stride);       // size
                htodstream->record(channel_job_next.inputEnd);
            }

            // Wait for current batch to finish
            executestream->synchronize();

            // Add input and preprocessing time for the current channel job
            input_timer->Add(channel_job.inputEnd.elapsedTime(channel_job.inputStart));
            preprocessing_timer->Add(channel_job.preprocessingEnd.elapsedTime(channel_job.preprocessingStart));

            // Add dedispersion time for current dm jobs
            for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
            {
                unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
                if (dm_job_id >= dm_jobs.size())
                {
                    break;
                }
                auto& dm_job = dm_jobs[dm_job_id];
                dedispersion_timer->Add(dm_job.dedispersionEnd.elapsedTime(dm_job.dedispersionStart));
            }
        } // end for ichan_start

        // Output DM batches
        for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
        {
            unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
            if (dm_job_id >= dm_jobs.size())
            {
                break;
            }
            auto& dm_job = dm_jobs[dm_job_id];

            // Get pointer to DM output data on host and on device
            dedisp_size dm_stride = nsamp_padded * out_bytes_per_sample;
            dedisp_size dm_offset = dm_job.idm_start * dm_stride;
            auto* h_data_t_dm_ptr = (void *) (((size_t) h_data_t_dm.data()) + dm_offset);
            auto* d_data_f_dm_ptr = (float *) dm_job.d_data_f_dm_ptr;
            auto* d_data_t_dm_ptr = (float *) dm_job.d_data_t_dm_ptr;

            // Fourier transform results back to time domain
            executestream->record(dm_job.postprocessingStart);
            for (unsigned int idm = 0; idm < dm_job.ndm_current; idm++)
            {
                auto *idata = (cufftComplex *) d_data_f_dm_ptr + (1ULL * idm * nsamp_padded/2);
                auto *odata = (cufftReal *) d_data_t_dm_ptr + (1ULL * idm * nsamp_padded);
                cufftExecC2R(plan_c2r, idata, odata);
            }

            // FFT scaling
            kernel.scale(
                dm_job.ndm_current, // height
                nsamp_padded,       // width
                nsamp_padded,       // stride
                1.0f / nfft,        // scale
                d_data_t_dm_ptr,    // d_data
                *executestream);    // stream
            executestream->record(dm_job.postprocessingEnd);

            // Copy output
            dtohstream->waitEvent(dm_job.postprocessingEnd);
            dtohstream->record(dm_job.outputStart);
            dtohstream->memcpyDtoHAsync(
                h_data_t_dm_ptr,                 // dst
                d_data_t_dm_ptr,                 // src
                dm_job.ndm_current * dm_stride); // size
            dtohstream->record(dm_job.outputEnd);
        } // end for dm_job_id_inner
    } // end for dm_job_id_outer

    // Wait for final memory transfer
    dtohstream->record(eEndGPU);
    mExeGPU.end(eEndGPU);
    dtohstream->synchronize();

    // Copy output
    std::cout << copy_output_str << std::endl;
    mCopyMem.start();
    output_timer->Start();
    copy_chunk_output(
        (float *) h_data_t_dm.data(), (float *) out,
        ndm, nsamp, nsamp_computed,
        nsamp_padded, nsamp_good, chunks);
    output_timer->Pause();
    mCopyMem.end();
    total_timer->Pause();

    // Accumulate dedispersion and postprocessing time for all dm jobs
    for (auto& job : dm_jobs)
    {
        postprocessing_timer->Add(job.postprocessingEnd.elapsedTime(job.postprocessingStart));
    }

    // Print timings
    long double runtime_time = preprocessing_timer->Milliseconds() + dedispersion_timer->Milliseconds() + postprocessing_timer->Milliseconds();
    runtime_time *= 1e-3; //seconds
    std::stringstream runtime_time_string;
    runtime_time_string << std::fixed;
    runtime_time_string << runtime_time;

    std::cout << timings_str << std::endl;
    std::cout << init_time_str           << init_timer->ToString() << " sec." << std::endl;
    std::cout << input_memcpy_time_str   << input_timer->ToString() << " sec." << std::endl;
    std::cout << preprocessing_time_str  << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << dedispersion_time_str   << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << postprocessing_time_str << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << output_memcpy_time_str  << output_timer->ToString() << " sec." << std::endl;
    std::cout << runtime_time_str        << runtime_time_string.str() << " sec." << std::endl;
    std::cout << total_time_str          << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}

// Private helper functions
void FDDGPUPlan::generate_spin_frequency_table(
    dedisp_size nfreq,
    dedisp_size nsamp,
    dedisp_float dt)
{
    h_spin_frequencies.resize(nfreq);

    #pragma omp parallel for
    for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
    {
        h_spin_frequencies[ifreq] = ifreq * (1.0/(nsamp*dt));
    }

    d_spin_frequencies.resize(nfreq * sizeof(dedisp_float));

    htodstream->memcpyHtoDAsync(d_spin_frequencies, h_spin_frequencies.data(), d_spin_frequencies.size());
}

} // end namespace dedisp