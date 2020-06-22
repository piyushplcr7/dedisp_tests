#include <cmath>

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
    cu::DeviceMemory d_in(in_count_gulp_max * sizeof(dedisp_word));
    cu::DeviceMemory d_transposed(in_count_padded_gulp_max * sizeof(dedisp_word));
    cu::DeviceMemory d_unpacked(unpacked_count_padded_gulp_max * sizeof(dedisp_word));
    cu::DeviceMemory d_out(out_count_gulp_max * sizeof(dedisp_word));

    // Organise host memory pointers
    cu::HostMemory h_in(in_count_gulp_max * sizeof(dedisp_word));
    cu::HostMemory h_out(out_count_gulp_max * sizeof(dedisp_word));

#ifdef DEDISP_BENCHMARK
    std::unique_ptr<Stopwatch> copy_to_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> copy_from_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> transpose_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> kernel_timer(Stopwatch::create());
#endif

    // Gulp loop
    for( dedisp_size gulp_samp_idx=0;
         gulp_samp_idx<nsamps_computed;
         gulp_samp_idx+=nsamps_computed_gulp_max ) {

        dedisp_size nsamps_computed_gulp = std::min(nsamps_computed_gulp_max,
                                               nsamps_computed-gulp_samp_idx);
        dedisp_size nsamps_gulp          = nsamps_computed_gulp + m_max_delay;
        dedisp_size nsamps_padded_gulp   = div_round_up(nsamps_computed_gulp,
                                                        DEDISP_SAMPS_PER_THREAD)
            * DEDISP_SAMPS_PER_THREAD + m_max_delay;

#ifdef DEDISP_BENCHMARK
        copy_to_timer->Start();
#endif
        // Copy the input data from host to device
        htodstream.memcpyHtoH2DAsync(
            h_in,                                 // dst
            in_buf_stride_words * BYTES_PER_WORD, // dst stride
            in + gulp_samp_idx*in_stride,         // src
            in_stride,                            // src stride
            nchan_words * BYTES_PER_WORD,         // width bytes
            nsamps_gulp);                         // height
        htodstream.memcpyHtoDAsync(
            d_in, // dst
            h_in, // src
            nchan_words * nsamps_gulp * BYTES_PER_WORD);
        htodstream.synchronize();
#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
        copy_to_timer->Pause();
        transpose_timer->Start();
#endif
        // Transpose the words in the input
        transpose((dedisp_word *) d_in,
                  nchan_words, nsamps_gulp,
                  in_buf_stride_words, nsamps_padded_gulp,
                  (dedisp_word *) d_transposed);
#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
        transpose_timer->Pause();

        kernel_timer->Start();
#endif

        // Unpack the transposed data
        unpack(d_transposed, nsamps_padded_gulp, nchan_words,
               d_unpacked,
               in_nbits, unpacked_in_nbits);

        // Perform direct dedispersion without scrunching
        if( !dedisperse(//d_transposed,
                        d_unpacked,
                        nsamps_padded_gulp,
                        nsamps_computed_gulp,
                        unpacked_in_nbits, //in_nbits,
                        m_nchans,
                        1,
                        d_dm_list,
                        dm_count,
                        1,
                        d_out,
                        out_stride_gulp_samples,
                        out_nbits,
                        1, 0, 0, 0, 0) ) {
            throw_error(DEDISP_INTERNAL_GPU_ERROR);
        }

#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
        kernel_timer->Pause();
#endif
        // Copy output back to host memory
        dedisp_size gulp_samp_byte_idx = gulp_samp_idx * out_bytes_per_sample;
        dedisp_size nsamp_bytes_computed_gulp = nsamps_computed_gulp * out_bytes_per_sample;
#ifdef DEDISP_BENCHMARK
        copy_from_timer->Start();
#endif
        dtohstream.memcpyDtoHAsync(
            h_out, // dst
            d_out, // src
            out_count_gulp_max);
        dtohstream.memcpyHtoH2DAsync(
            out + gulp_samp_byte_idx,  // dst
            out_stride,                // dst stride
            (byte_type *) d_out,       // src
            out_stride_gulp_bytes,     // src stride
            nsamp_bytes_computed_gulp, // width bytes
            dm_count);                 // height
        dtohstream.synchronize();
#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
        copy_from_timer->Pause();
#endif

    } // End of gulp loop

#ifdef DEDISP_BENCHMARK
    cout << "Copy to time:   " << copy_to_timer->ToString() << endl;
    cout << "Copy from time: " << copy_from_timer->ToString() << endl;
    cout << "Transpose time: " << transpose_timer->ToString() << endl;
    cout << "Kernel time:    " << kernel_timer->ToString() << endl;
    auto total_time = copy_to_timer->Milliseconds() +
                      copy_from_timer->Milliseconds() +
                      transpose_timer->Milliseconds() +
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