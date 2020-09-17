#include <thread>
#include <mutex>
#include <iostream>

#include "TDDPlan.hpp"

#include "common/dedisp_strings.h"
#include "common/cuda/CU.h"
#include "common/helper.h"

#include "dedisperse/TDDKernel.hpp"
#include "unpack/unpack.h"

#include "dedisp_error.hpp"

#if defined(DEDISP_BENCHMARK)
#include <fstream>
#include "external/Stopwatch.h"
#endif

namespace dedisp
{

// Constructor
TDDPlan::TDDPlan(
    size_type  nchans,
    float_type dt,
    float_type f0,
    float_type df) :
    GPUPlan(nchans, dt, f0, df)
{
    // Check for parameter errors
    if( nchans > compute_max_nchans() ) {
        throw_error(DEDISP_NCHANS_EXCEEDS_LIMIT);
    }

    m_gulp_size = compute_gulp_size();

    // Initialize kernel
    initialize_kernel();
}

// Destructor
TDDPlan::~TDDPlan()
{}

// Private helper functions
unsigned long div_round_up(unsigned long a, unsigned long b) {
    return (a-1) / b + 1;
}

dedisp_size TDDPlan::compute_gulp_size()
{
    return 65536;
}

dedisp_size TDDPlan::compute_max_nchans()
{
    size_t const_mem_bytes = m_device->get_total_const_memory();
    size_t bytes_per_chan = sizeof(dedisp_float) + sizeof(dedisp_bool);
    size_t max_nr_channels = const_mem_bytes / bytes_per_chan;
    return max_nr_channels;
};

void TDDPlan::initialize_kernel()
{
    // Configure texture memory based on compute capability
    auto capability = m_device->get_capability();
    if (capability < 20 ||                    // Pre Fermi
        capability == 60 || capability == 61) // Pascal
    {
        m_kernel.use_texture_memory(true);
    }
}

// Public interface
void TDDPlan::set_gulp_size(
    size_type gulp_size)
{
    m_gulp_size = gulp_size;
}

void TDDPlan::execute(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits)
{
    enum {
        BITS_PER_BYTE = 8
    };

    // Note: The default out_stride is nsamps - m_max_delay
    dedisp_size out_bytes_per_sample =
        out_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);

    // Note: Must be careful with integer division
    dedisp_size in_stride =
        m_nchans * in_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);
    dedisp_size out_stride = (nsamps - m_max_delay) * out_bytes_per_sample;

    execute_adv(
        nsamps,
        in, in_nbits, in_stride,
        out, out_nbits, out_stride);
}

void TDDPlan::execute_adv(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    size_type        in_stride,
    byte_type*       out,
    size_type        out_nbits,
    size_type        out_stride)
{
    dedisp_size first_dm_idx = 0;
    dedisp_size dm_count = m_dm_count;

    execute_guru(
        nsamps,
        in, in_nbits, in_stride,
        out, out_nbits, out_stride,
        first_dm_idx, dm_count);
}

void TDDPlan::execute_guru(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    size_type        in_stride,
    byte_type*       out,
    size_type        out_nbits,
    size_type        out_stride,
    size_type        first_dm_idx,
    size_type        dm_count)
{
    enum {
        BITS_PER_BYTE  = 8,
        BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
    };

    dedisp_size out_bytes_per_sample = out_nbits / (sizeof(dedisp_byte) *
                                                    BITS_PER_BYTE);

    if( 0 == in || 0 == out ) {
        throw_error(DEDISP_INVALID_POINTER);
    }
    // Note: Must be careful with integer division
    if( in_stride < m_nchans*in_nbits/(sizeof(dedisp_byte)*BITS_PER_BYTE) ||
        out_stride < (nsamps - m_max_delay)*out_bytes_per_sample ) {
        throw_error(DEDISP_INVALID_STRIDE);
    }
    if( 0 == m_dm_count ) {
        throw_error(DEDISP_NO_DM_LIST_SET);
    }
    if( nsamps < m_max_delay ) {
        throw_error(DEDISP_TOO_FEW_NSAMPS);
    }

    // Check for valid nbits values
    if( in_nbits  != 1 &&
        in_nbits  != 2 &&
        in_nbits  != 4 &&
        in_nbits  != 8 &&
        in_nbits  != 16 &&
        in_nbits  != 32 ) {
        throw_error(DEDISP_UNSUPPORTED_IN_NBITS);
    }
    if( out_nbits != 8 &&
        out_nbits != 16 &&
        out_nbits != 32 ) {
        throw_error(DEDISP_UNSUPPORTED_OUT_NBITS);
    }

    // Timers
#ifdef DEDISP_BENCHMARK
    std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> input_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();
#endif
    // Annotate the initialization
    cu::Marker initMarker("initialization", cu::Marker::red);
    initMarker.start();

    // Copy the lookup tables to constant memory on the device
    cu::Marker constantMarker("copy_constant_memory", cu::Marker::yellow);
    constantMarker.start();
    m_kernel.copy_delay_table(
        d_delay_table,
        m_nchans * sizeof(dedisp_float),
        0, *htodstream);
    m_kernel.copy_killmask(
        d_killmask,
        m_nchans * sizeof(dedisp_bool),
        0, *htodstream);
    constantMarker.end();

    // Compute the problem decomposition
    dedisp_size nsamps_computed = nsamps - m_max_delay;

    // Specify the maximum gulp size
    dedisp_size nsamps_computed_gulp_max = std::min(m_gulp_size, nsamps_computed);

    // Compute derived counts for maximum gulp size [dedisp_word == 4 bytes]
    dedisp_size nsamps_gulp_max = nsamps_computed_gulp_max + m_max_delay;
    dedisp_size chans_per_word  = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;
    dedisp_size nchan_words     = m_nchans / chans_per_word;

    // Note: If desired, this could be rounded up, e.g., to a power of 2
    dedisp_size in_buf_stride_words      = nchan_words;
    dedisp_size in_count_gulp_max        = nsamps_gulp_max * in_buf_stride_words;
    dedisp_size samps_per_thread         = m_kernel.get_nsamps_per_thread();

    dedisp_size nsamps_padded_gulp_max   = div_round_up(nsamps_computed_gulp_max,
                                                        samps_per_thread)
                                                      * samps_per_thread + m_max_delay;
    dedisp_size in_count_padded_gulp_max =
        nsamps_padded_gulp_max * in_buf_stride_words;

    // TODO: Make this a parameter?
    dedisp_size min_in_nbits = 0;
    dedisp_size unpacked_in_nbits = std::max((int)in_nbits, (int)min_in_nbits);
    dedisp_size unpacked_chans_per_word =
        sizeof(dedisp_word)*BITS_PER_BYTE / unpacked_in_nbits;
    dedisp_size unpacked_nchan_words = m_nchans / unpacked_chans_per_word;
    dedisp_size unpacked_buf_stride_words = unpacked_nchan_words;
    dedisp_size unpacked_count_padded_gulp_max =
        nsamps_padded_gulp_max * unpacked_buf_stride_words;

    dedisp_size out_stride_gulp_samples  = nsamps_computed_gulp_max;
    dedisp_size out_stride_gulp_bytes    =
        out_stride_gulp_samples * out_bytes_per_sample;
    dedisp_size out_count_gulp_max       = out_stride_gulp_bytes * dm_count;

    // Compute the number of gulps (jobs)
    unsigned int nr_gulps = div_round_up(nsamps_computed, nsamps_computed_gulp_max);

    // Organise device memory pointers
    std::cout << memory_alloc_str << std::endl;
    cu::DeviceMemory d_transposed(in_count_padded_gulp_max * sizeof(dedisp_word));
    cu::DeviceMemory d_unpacked(unpacked_count_padded_gulp_max * sizeof(dedisp_word));
    cu::DeviceMemory d_out(out_count_gulp_max * sizeof(dedisp_word));
    cu::HostMemory h_out(out_count_gulp_max * sizeof(dedisp_word));
    std::vector<cu::HostMemory> h_in_(2);
    std::vector<cu::DeviceMemory> d_in_(2);
    for (unsigned int i = 0; i < 2; i++)
    {
        h_in_[i].resize(in_count_gulp_max * sizeof(dedisp_word));
        d_in_[i].resize(in_count_gulp_max * sizeof(dedisp_word));
    }

    struct JobData {
        dedisp_size gulp_samp_idx;
        dedisp_size nsamps_computed_gulp;
        dedisp_size nsamps_gulp;
        dedisp_size nsamps_padded_gulp;
        void *h_in_ptr;
        void *d_in_ptr;
        std::mutex output_lock;
        cu::Event inputStart, inputEnd;
        cu::Event preprocessingStart, preprocessingEnd;
        cu::Event dedispersionStart, dedispersionEnd;
        cu::Event outputStart, outputEnd;
    };

    std::vector<JobData> jobs(nr_gulps);

    for (unsigned int gulp = 0; gulp < nr_gulps; gulp++)
    {
        JobData& job = jobs[gulp];
        job.gulp_samp_idx        = gulp == 0 ? 0 : jobs[gulp - 1].gulp_samp_idx
                                   + nsamps_computed_gulp_max;
        job.nsamps_computed_gulp = std::min(nsamps_computed_gulp_max,
                                   nsamps_computed-job.gulp_samp_idx);
        job.nsamps_gulp          = job.nsamps_computed_gulp + m_max_delay;
        job.nsamps_padded_gulp   = div_round_up(job.nsamps_computed_gulp,
                                                samps_per_thread)
                                              * samps_per_thread + m_max_delay;
        job.h_in_ptr             = h_in_[gulp % 2];
        job.d_in_ptr             = d_in_[gulp % 2];
        job.output_lock.lock();
    }

    std::thread output_thread = std::thread([&]()
    {
        for (auto& job : jobs)
        {
            // Wait for the GPU to finish
            job.output_lock.lock();
            job.outputEnd.synchronize();

            dedisp_size gulp_samp_byte_idx = job.gulp_samp_idx * out_bytes_per_sample;
            dedisp_size nsamp_bytes_computed_gulp = job.nsamps_computed_gulp * out_bytes_per_sample;

            // Copy output
            memcpy2D(
                out + gulp_samp_byte_idx,  // dst
                out_stride,                // dst stride
                (byte_type *) h_out,       // src
                out_stride_gulp_bytes,     // src stride
                nsamp_bytes_computed_gulp, // width bytes
                dm_count);                 // height
        }
    });

    // The initialization is finished
#ifdef DEDISP_BENCHMARK
    init_timer->Pause();
#endif
    initMarker.end();

    // Annotate the gulp loop
    cu::ScopedMarker gulpMarker("gulp_loop", cu::Marker::black);

#ifdef DEDISP_BENCHMARK
    // Measure the total time of the gulp loop
    cu::Event gulpStart, gulpEnd;
    htodstream->record(gulpStart);
#endif

    // Gulp loop
    std::cout << ref_dedispersion_str << std::endl;
    for (unsigned job_id = 0; job_id < jobs.size(); job_id++)
    {
        // Wait for previous job to finish to
        // prevent overwriting its input buffer
        if (job_id > 1)
        {
            auto& job_previous = jobs[job_id - 2];
            job_previous.dedispersionEnd.synchronize();
        }

        auto& job = jobs[job_id];

        // Copy the input data for the current job
        memcpy2D(
            job.h_in_ptr,                         // dst
            in_buf_stride_words * BYTES_PER_WORD, // dst stride
            in + job.gulp_samp_idx*in_stride,     // src
            in_stride,                            // src stride
            nchan_words * BYTES_PER_WORD,         // width bytes
            job.nsamps_gulp);                     // height
        htodstream->record(job.inputStart);
        htodstream->memcpyHtoDAsync(
            job.d_in_ptr,                                    // dst
            job.h_in_ptr,                                    // src
            nchan_words * job.nsamps_gulp * BYTES_PER_WORD); // size
        htodstream->record(job.inputEnd);

        // Transpose and unpack the words in the input
        executestream->waitEvent(job.inputEnd);
        executestream->record(job.preprocessingStart);
        transpose_unpack(
            (dedisp_word *) job.d_in_ptr,
            nchan_words, job.nsamps_gulp,
            in_buf_stride_words, job.nsamps_padded_gulp,
            (dedisp_word *) d_unpacked,
            in_nbits, unpacked_in_nbits,
            *executestream);
        executestream->record(job.preprocessingEnd);

        // Perform direct dedispersion without scrunching
        executestream->record(job.dedispersionStart);
        m_kernel.launch(
            d_unpacked,               // d_in
            job.nsamps_padded_gulp,   // in_stride
            job.nsamps_computed_gulp, // nsamps
            unpacked_in_nbits,        // in_nbits,
            m_nchans,                 // nchans
            1,                        // chan_stride
            d_dm_list,                // d_dm_list
            dm_count,                 // dm_count
            1,                        // dm_stride
            d_out,                    // d_out
            out_stride_gulp_samples,  // out_stride
            out_nbits,                // out_nbits
            *executestream);
        executestream->record(job.dedispersionEnd);

        // Copy output back to host memory
        dtohstream->waitEvent(job.dedispersionEnd);
        dtohstream->record(job.outputStart);
        dtohstream->memcpyDtoHAsync(
            h_out,               // dst
            d_out,               // src
            out_count_gulp_max); // size
        dtohstream->record(job.outputEnd);
        job.output_lock.unlock();
    } // End of gulp loop

#ifdef DEDISP_BENCHMARK
    dtohstream->record(gulpEnd);
    gulpEnd.synchronize();
    total_timer->Pause();

    // Accumulate dedispersion and memcopy time for all jobs
    for (auto& job : jobs)
    {
        input_timer->Add(job.inputEnd.elapsedTime(job.inputStart));
        output_timer->Add(job.outputEnd.elapsedTime(job.outputStart));
        preprocessing_timer->Add(job.preprocessingEnd.elapsedTime(job.preprocessingStart));
        dedispersion_timer->Add(job.dedispersionEnd.elapsedTime(job.dedispersionStart));
    }

    // Print timings
    long double runtime_time = preprocessing_timer->Milliseconds() + dedispersion_timer->Milliseconds();
    runtime_time *= 10e-3; //seconds
    std::stringstream runtime_time_string;
    runtime_time_string << std::fixed;
    runtime_time_string << runtime_time;

    std::cout << timings_str << std::endl;
    std::cout << init_time_str           << init_timer->ToString() << " sec." << std::endl;
    std::cout << preprocessing_time_str  << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << dedispersion_time_str   << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << input_memcpy_time_str   << input_timer->ToString() << " sec." << std::endl;
    std::cout << output_memcpy_time_str  << output_timer->ToString() << " sec." << std::endl;
    std::cout << runtime_time_str        << runtime_time_string.str() << " sec." << std::endl;
    std::cout << total_time_str          << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;

    // Compute number of operations performed
    unsigned long dedispersion_ops = 0;
    unsigned long preprocessing_ops = 0;
    for (auto& job : jobs)
    {
        unsigned long nsamp_processed_gulp = job.nsamps_computed_gulp * m_nchans;
        // dm * frac_delay, only once for samps_per_thread samples
        dedispersion_ops += m_dm_count * nsamp_processed_gulp * (1.0/samps_per_thread);
        // sum += killmask * sample, fma
        dedispersion_ops += m_dm_count * nsamp_processed_gulp * 2;
        // construct output, +/- 5 integer operations
        preprocessing_ops += nsamp_processed_gulp * 5;
    }

    // Print performance
    auto preprocessing_performance = 1e-6 * preprocessing_ops / preprocessing_timer->Milliseconds();
    auto dedispersion_performance = 1e-6 * dedispersion_ops / dedispersion_timer->Milliseconds();
    std::cout << preprocessing_perf_str << preprocessing_performance << " GOps/s" << std::endl;
    std::cout << dedispersion_perf_str << dedispersion_performance << " GOps/s" << std::endl;

    // Append the timing results to a log file
    auto total_time = Stopwatch::ToString(gulpEnd.elapsedTime(gulpStart));
    std::ofstream perf_file("perf.log", std::ios::app);
    perf_file << input_timer->ToString() << "\t"
              << output_timer->ToString() << "\t"
              << dedispersion_timer->ToString() << "\t"
              << total_time << std::endl;
    perf_file.close();
#endif

    // Wait for host threads to exit
    if (output_thread.joinable()) { output_thread.join(); }
}

} // end namespace dedisp