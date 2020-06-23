#include <cmath>
#include <cstring>
#include <memory>
#include <iostream>
#include <thread>
#include <mutex>

#include <cuda_runtime.h>

#include "DedispPlan.hpp"
#include "transpose/transpose.hpp"

#include "dedisp_defines.h"
#include "dedisp_error.hpp"
#include "dedisperse/dedisperse.h"
#include "unpack/unpack.h"
#include "common/cuda/CU.h"

#if defined(DEDISP_BENCHMARK)
#include <iostream>
#include <fstream>
#include <memory>
using std::cout;
using std::endl;
#include "external/Stopwatch.h"
#endif

namespace dedisp
{

// Constructor
DedispPlan::DedispPlan(size_type  nchans,
                       float_type dt,
                       float_type f0,
                       float_type df)
{
    cu::checkError();

    int device_idx;
    cudaGetDevice(&device_idx);

    // Check for parameter errors
    if( nchans > DEDISP_MAX_NCHANS ) {
        throw_error(DEDISP_NCHANS_EXCEEDS_LIMIT);
    }

    // Force the df parameter to be negative such that
    //   freq[chan] = f0 + chan * df.
    df = -std::abs(df);

    m_dm_count      = 0;
    m_nchans        = nchans;
    m_gulp_size     = DEDISP_DEFAULT_GULP_SIZE;
    m_max_delay     = 0;
    m_dt            = dt;
    m_f0            = f0;
    m_df            = df;

    // Generate delay table and copy to device memory
    // Note: The DM factor is left out and applied during dedispersion
    h_delay_table.resize(nchans);
    generate_delay_table(h_delay_table.data(), nchans, dt, f0, df);
    d_delay_table.resize(nchans * sizeof(dedisp_float));
    htodstream.memcpyHtoDAsync(d_delay_table, h_delay_table.data(), d_delay_table.size());

    // Initialise the killmask
    h_killmask.resize(nchans, (dedisp_bool)true);
    d_killmask.resize(nchans * sizeof(dedisp_bool));
    set_killmask((dedisp_bool*)0);

    htodstream.synchronize();
}

// Destructor
DedispPlan::~DedispPlan() {
}

// Private helper functions
unsigned long div_round_up(unsigned long a, unsigned long b) {
    return (a-1) / b + 1;
}

void DedispPlan::generate_delay_table(dedisp_float* h_delay_table, dedisp_size nchans,
                                      dedisp_float dt, dedisp_float f0, dedisp_float df)
{
    for( dedisp_size c=0; c<nchans; ++c ) {
        dedisp_float a = 1.f / (f0+c*df);
        dedisp_float b = 1.f / f0;
        // Note: To higher precision, the constant is 4.148741601e3
        h_delay_table[c] = 4.15e3/dt * (a*a - b*b);
    }
}

dedisp_float DedispPlan::get_smearing(dedisp_float dt, dedisp_float pulse_width,
                                      dedisp_float f0, dedisp_size nchans, dedisp_float df,
                                      dedisp_float DM, dedisp_float deltaDM)
{
    dedisp_float W         = pulse_width;
    dedisp_float BW        = nchans * abs(df);
    dedisp_float fc        = f0 - BW/2;
    dedisp_float inv_fc3   = 1./(fc*fc*fc);
    dedisp_float t_DM      = 8.3*BW*DM*inv_fc3;
    dedisp_float t_deltaDM = 8.3/4*BW*nchans*deltaDM*inv_fc3;
    dedisp_float t_smear   = std::sqrt(dt*dt + W*W + t_DM*t_DM + t_deltaDM*t_deltaDM);
    return t_smear;
}

void DedispPlan::generate_dm_list(std::vector<dedisp_float>& dm_table,
                                  dedisp_float dm_start, dedisp_float dm_end,
                                  double dt, double ti, double f0, double df,
                                  dedisp_size nchans, double tol)
{
    // Note: This algorithm originates from Lina Levin
    // Note: Computation done in double precision to match MB's code

    dt *= 1e6;
    double f    = (f0 + ((nchans/2) - 0.5) * df) * 1e-3;
    double tol2 = tol*tol;
    double a    = 8.3 * df / (f*f*f);
    double a2   = a*a;
    double b2   = a2 * (double)(nchans*nchans / 16.0);
    double c    = (dt*dt + ti*ti) * (tol2 - 1.0);

    dm_table.push_back(dm_start);
    while( dm_table.back() < dm_end ) {
        double prev     = dm_table.back();
        double prev2    = prev*prev;
        double k        = c + tol2*a2*prev2;
        double dm = ((b2*prev + std::sqrt(-a2*b2*prev2 + (a2+b2)*k)) / (a2+b2));
        dm_table.push_back(dm);
    }
}

void DedispPlan::memcpy2D(
    void *dstPtr, size_t dstWidth,
    const void *srcPtr, size_t srcWidth,
    size_t widthBytes, size_t height)
{
    typedef char SrcType[height][srcWidth];
    typedef char DstType[height][dstWidth];
    auto src = (SrcType *) srcPtr;
    auto dst = (DstType *) dstPtr;
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < widthBytes; x++)
        {
            (*dst)[y][x] = (*src)[y][x];
        }
    }
}

// Public interface
void DedispPlan::set_device(int device_idx) {
    cu::Device device(device_idx);
}

void DedispPlan::set_gulp_size(size_type gulp_size) {
    m_gulp_size = gulp_size;
}

void DedispPlan::set_killmask(const bool_type* killmask)
{
    cu::checkError();

    if( 0 != killmask ) {
        // Copy killmask to plan (both host and device)
        h_killmask.assign(killmask, killmask + m_nchans);
        htodstream.memcpyHtoDAsync(d_killmask, h_killmask.data(), d_killmask.size());
    }
    else {
        // Set the killmask to all true
        std::fill(h_killmask.begin(), h_killmask.end(), (dedisp_bool)true);
        htodstream.memcpyHtoDAsync(d_killmask, h_killmask.data(), d_killmask.size());
    }
}

void DedispPlan::set_dm_list(const float_type* dm_list,
                             size_type         count)
{
    if( !dm_list ) {
        throw_error(DEDISP_INVALID_POINTER);
    }
    cu::checkError();

    m_dm_count = count;
    h_dm_list.assign(dm_list, dm_list+count);

    // Copy to the device
    d_dm_list.resize(m_dm_count * sizeof(dedisp_float));
    htodstream.memcpyHtoDAsync(d_dm_list, h_dm_list.data(), d_dm_list.size());
    htodstream.synchronize();

    // Calculate the maximum delay and store it in the plan
    m_max_delay = dedisp_size(h_dm_list[m_dm_count-1] *
                              h_delay_table[m_nchans-1] + 0.5);
}

void DedispPlan::generate_dm_list(float_type dm_start,
                                  float_type dm_end,
                                  float_type ti,
                                  float_type tol)
{
    cu::checkError();

    // Generate the DM list (on the host)
    h_dm_list.clear();
    generate_dm_list(
        h_dm_list,
        dm_start, dm_end,
        m_dt, ti, m_f0, m_df,
        m_nchans, tol);
    m_dm_count = h_dm_list.size();

    // Allocate device memory for the DM list
    d_dm_list.resize(m_dm_count * sizeof(dedisp_float));
    htodstream.memcpyHtoDAsync(d_dm_list, h_dm_list.data(), d_dm_list.size());
    htodstream.synchronize();

    // Calculate the maximum delay and store it in the plan
    m_max_delay = dedisp_size(h_dm_list[m_dm_count-1] *
                              h_delay_table[m_nchans-1] + 0.5);
}

void DedispPlan::execute(size_type        nsamps,
                         const byte_type* in,
                         size_type        in_nbits,
                         byte_type*       out,
                         size_type        out_nbits,
                         unsigned         flags)
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
        out, out_nbits, out_stride,
        flags);
}

void DedispPlan::execute_adv(size_type        nsamps,
                             const byte_type* in,
                             size_type        in_nbits,
                             size_type        in_stride,
                             byte_type*       out,
                             size_type        out_nbits,
                             size_type        out_stride,
                             unsigned         flags)
{
    dedisp_size first_dm_idx = 0;
    dedisp_size dm_count = m_dm_count;

    execute_guru(
        nsamps,
        in, in_nbits, in_stride,
        out, out_nbits, out_stride,
        first_dm_idx, dm_count,
        flags);
}

void DedispPlan::execute_guru(size_type        nsamps,
                              const byte_type* in,
                              size_type        in_nbits,
                              size_type        in_stride,
                              byte_type*       out,
                              size_type        out_nbits,
                              size_type        out_stride,
                              size_type        first_dm_idx,
                              size_type        dm_count,
                              unsigned         flags)
{
    cu::checkError();

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

    // Check for valid synchronisation flags
    if( flags & DEDISP_ASYNC && flags & DEDISP_WAIT ) {
        throw_error(DEDISP_INVALID_FLAG_COMBINATION);
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

    // Copy the lookup tables to constant memory on the device
    copy_delay_table(d_delay_table,
                     m_nchans * sizeof(dedisp_float),
                     0, 0);
    copy_killmask(d_killmask,
                  m_nchans * sizeof(dedisp_bool),
                  0, 0);

    // Compute the problem decomposition
    dedisp_size nsamps_computed = nsamps - m_max_delay;

    // Specify the maximum gulp size
    dedisp_size nsamps_computed_gulp_max = std::min(m_gulp_size, nsamps_computed);

    // Just to be sure
    // TODO: This seems quite wrong. Why was it here?
    /*
    if( nsamps_computed_gulp_max < m_max_delay ) {
        throw_error(DEDISP_TOO_FEW_NSAMPS);
    }
    */

    // Compute derived counts for maximum gulp size [dedisp_word == 4 bytes]
    dedisp_size nsamps_gulp_max = nsamps_computed_gulp_max + m_max_delay;
    dedisp_size chans_per_word  = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;
    dedisp_size nchan_words     = m_nchans / chans_per_word;

    // We use words for processing but allow arbitrary byte strides, which are
    //   not necessarily friendly.
    //bool friendly_in_stride = (0 == in_stride % BYTES_PER_WORD);

    // Note: If desired, this could be rounded up, e.g., to a power of 2
    dedisp_size in_buf_stride_words      = nchan_words;
    dedisp_size in_count_gulp_max        = nsamps_gulp_max * in_buf_stride_words;

    dedisp_size nsamps_padded_gulp_max   = div_round_up(nsamps_computed_gulp_max,
                                                        DEDISP_SAMPS_PER_THREAD)
        * DEDISP_SAMPS_PER_THREAD + m_max_delay;
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

    // Organise device memory pointers
    cu::DeviceMemory d_transposed(in_count_padded_gulp_max * sizeof(dedisp_word));
    cu::DeviceMemory d_unpacked(unpacked_count_padded_gulp_max * sizeof(dedisp_word));
    cu::DeviceMemory d_out(out_count_gulp_max * sizeof(dedisp_word));

    // Two input buffers for double buffering
    std::vector<cu::HostMemory> h_in_(2);
    std::vector<cu::DeviceMemory> d_in_(2);
    for (unsigned int i = 0; i < 2; i++)
    {
        h_in_[i].resize(in_count_gulp_max * sizeof(dedisp_word));
        d_in_[i].resize(in_count_gulp_max * sizeof(dedisp_word));
    }

    // Organise host memory pointers
    cu::HostMemory h_out(out_count_gulp_max * sizeof(dedisp_word));

    // Compute the number of gulps (jobs)
    unsigned int nr_gulps = div_round_up(nsamps_computed, nsamps_computed_gulp_max);

#ifdef DEDISP_BENCHMARK
    std::unique_ptr<Stopwatch> copy_to_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> copy_from_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> transpose_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> kernel_timer(Stopwatch::create());
#endif

    struct JobData {
        dedisp_size gulp_samp_idx;
        dedisp_size nsamps_computed_gulp;
        dedisp_size nsamps_gulp;
        dedisp_size nsamps_padded_gulp;
        void *h_in_ptr;
        void *d_in_ptr;
        std::mutex lockCPU, lockGPU;
        cu::Event inputStart, inputEnd;
        cu::Event computeStart, computeEnd;
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
                                    DEDISP_SAMPS_PER_THREAD)
                                  * DEDISP_SAMPS_PER_THREAD + m_max_delay;
        job.h_in_ptr             = h_in_[gulp % 2];
        job.d_in_ptr             = d_in_[gulp % 2];
        job.lockCPU.lock();
        job.lockGPU.lock();
    }

    std::thread input_thread = std::thread([&]()
    {
        for (auto& job : jobs)
        {
            // Copy input
            memcpy2D(
                job.h_in_ptr,                         // dst
                in_buf_stride_words * BYTES_PER_WORD, // dst stride
                in + job.gulp_samp_idx*in_stride,     // src
                in_stride,                            // src stride
                nchan_words * BYTES_PER_WORD,         // width bytes
                job.nsamps_gulp);                     // height

            // Signal that the job can start
            job.lockCPU.unlock();
        }
    });

    std::thread output_thread = std::thread([&]()
    {
        for (auto& job : jobs)
        {
            // Wait for the GPU to finish
            job.lockGPU.lock();
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

    // Gulp loop
    for (unsigned job_id = 0; job_id < jobs.size(); job_id++)
    {
        // Id for double buffering
        unsigned job_id_next   = job_id + 1;

        auto& job = jobs[job_id];

        // Copy the input data for the first job
        if (job_id == 0)
        {
            job.lockCPU.lock();
            htodstream.record(job.inputStart);
            htodstream.memcpyHtoDAsync(
                job.d_in_ptr, // dst
                job.h_in_ptr, // src
                nchan_words * job.nsamps_gulp * BYTES_PER_WORD);
            htodstream.record(job.inputEnd);
        }

        // Copy the input data for the next job
        if (job_id_next < jobs.size())
        {
            auto& job_next = jobs[job_id_next];
            job_next.lockCPU.lock();
            htodstream.record(job_next.inputStart);
            htodstream.memcpyHtoDAsync(
                job_next.d_in_ptr, // dst
                job_next.h_in_ptr, // src
                nchan_words * job.nsamps_gulp * BYTES_PER_WORD);
            htodstream.record(job_next.inputEnd);
        }

        // Transpose the words in the input
        executestream.waitEvent(job.inputEnd);
        executestream.record(job.computeStart);
        transpose((dedisp_word *) job.d_in_ptr,
                  nchan_words, job.nsamps_gulp,
                  in_buf_stride_words, job.nsamps_padded_gulp,
                  (dedisp_word *) d_transposed,
                  executestream);

        // Unpack the transposed data
        unpack(d_transposed, job.nsamps_padded_gulp, nchan_words,
               d_unpacked,
               in_nbits, unpacked_in_nbits,
               executestream);

        // Perform direct dedispersion without scrunching
        if( !dedisperse(//d_transposed,
                        d_unpacked,
                        job.nsamps_padded_gulp,
                        job.nsamps_computed_gulp,
                        unpacked_in_nbits, //in_nbits,
                        m_nchans,
                        1,
                        d_dm_list,
                        dm_count,
                        1,
                        d_out,
                        out_stride_gulp_samples,
                        out_nbits,
                        1, 0, 0, 0, 0,
                        executestream) ) {
            throw_error(DEDISP_INTERNAL_GPU_ERROR);
        }
        executestream.record(job.computeEnd);

        // Copy output back to host memory
        dtohstream.waitEvent(job.computeEnd);
        dtohstream.record(job.outputStart);
        dtohstream.memcpyDtoHAsync(
            h_out, // dst
            d_out, // src
            out_count_gulp_max);
        dtohstream.record(job.outputEnd);
        job.lockGPU.unlock();

    } // End of gulp loop

    if (input_thread.joinable()) { input_thread.join(); }
    if (output_thread.joinable()) { output_thread.join(); }

    dtohstream.synchronize();

#ifdef DEDISP_BENCHMARK
    for (auto& job : jobs)
    {
        copy_to_timer->Add(job.inputEnd.elapsedTime(job.inputStart));
        copy_from_timer->Add(job.outputEnd.elapsedTime(job.outputStart));
        kernel_timer->Add(job.computeEnd.elapsedTime(job.computeStart));
    }
#endif

#ifdef DEDISP_BENCHMARK
    cout << "Copy to time:   " << copy_to_timer->ToString() << endl;
    cout << "Copy from time: " << copy_from_timer->ToString() << endl;
    cout << "Kernel time:    " << kernel_timer->ToString() << endl;
    auto total_time = copy_to_timer->Milliseconds() +
                      copy_from_timer->Milliseconds() +
                      kernel_timer->Milliseconds();
    cout << "Total time:     " << Stopwatch::ToString(total_time) << endl;

    // Append the timing results to a log file
    std::ofstream perf_file("perf.log", std::ios::app);
    perf_file << copy_to_timer->ToString() << "\t"
              << copy_from_timer->ToString() << "\t"
              << transpose_timer->ToString() << "\t"
              << kernel_timer->ToString() << "\t"
              << Stopwatch::ToString(total_time) << endl;
    perf_file.close();
#endif
}

void DedispPlan::sync()
{
    cu::checkError(cudaDeviceSynchronize());
}

} // end namespace dedisp