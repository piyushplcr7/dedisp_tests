#include <cmath>

#include "DedispPlan.hpp"

#include "common/dedisp_types.h"
#include "common/dedisp_error.hpp"
#include "common/dedisp_strings.h"

#include "transpose/transpose.hpp"
#include "dedisperse/dedisperse.h"
#include "unpack/unpack.h"

#if defined(DEDISP_BENCHMARK)
#include <fstream>
#include <iostream>
#include <memory>
#include "external/Stopwatch.h"
#endif

namespace dedisp
{

// Constructor
DedispPlan::DedispPlan(size_type  nchans,
                       float_type dt,
                       float_type f0,
                       float_type df,
                       int device_idx) :
    GPUPlan(nchans, dt, f0, df, device_idx)
{
    // Check for parameter errors
    if( nchans > DEDISP_MAX_NCHANS) {
        throw_error(DEDISP_NCHANS_EXCEEDS_LIMIT);
    }

    m_gulp_size = DEDISP_DEFAULT_GULP_SIZE;
}

// Destructor
DedispPlan::~DedispPlan() {
}

// Private helper functions
unsigned long div_round_up(unsigned long a, unsigned long b) {
    return (a-1) / b + 1;
}

// Public interface
void DedispPlan::set_gulp_size(size_type gulp_size) {
    m_gulp_size = gulp_size;
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


#ifdef DEDISP_BENCHMARK
    // The initialization is finished
    init_timer->Pause();

    // Measure the total time of the gulp loop
    cu::Event gulpStart, gulpEnd;
    htodstream->record(gulpStart);
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
        input_timer->Start();
#endif
        // Copy the input data from host to device
        htodstream->memcpyHtoD2DAsync(
            d_in,                                 // dst
            in_buf_stride_words * BYTES_PER_WORD, // dst stride
            in + gulp_samp_idx*in_stride,         // src
            in_stride,                            // src stride
            nchan_words * BYTES_PER_WORD,         // width bytes
            nsamps_gulp);                         // height
        htodstream->synchronize();
#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
        input_timer->Pause();
        preprocessing_timer->Start();
#endif
        // Transpose the words in the input
        transpose((dedisp_word *) d_in,
                  nchan_words, nsamps_gulp,
                  in_buf_stride_words, nsamps_padded_gulp,
                  (dedisp_word *) d_transposed);
#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
#endif

        // Unpack the transposed data
        unpack(d_transposed, nsamps_padded_gulp, nchan_words,
               d_unpacked,
               in_nbits, unpacked_in_nbits);

#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
        preprocessing_timer->Pause();
        dedispersion_timer->Start();
#endif

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
        dedispersion_timer->Pause();
#endif
        // Copy output back to host memory
        dedisp_size gulp_samp_byte_idx = gulp_samp_idx * out_bytes_per_sample;
        dedisp_size nsamp_bytes_computed_gulp = nsamps_computed_gulp * out_bytes_per_sample;
#ifdef DEDISP_BENCHMARK
        output_timer->Start();
#endif
        dtohstream->memcpyDtoH2DAsync(
            out + gulp_samp_byte_idx,  // dst
            out_stride,                // dst stride
            (byte_type *) d_out,       // src
            out_stride_gulp_bytes,     // src stride
            nsamp_bytes_computed_gulp, // width bytes
            dm_count);                 // height
        dtohstream->synchronize();
#ifdef DEDISP_BENCHMARK
        cudaDeviceSynchronize();
        output_timer->Pause();
#endif

    } // End of gulp loop

#ifdef DEDISP_BENCHMARK
    dtohstream->record(gulpEnd);
    gulpEnd.synchronize();
    total_timer->Pause();

    // Print timings
    long double runtime_time = input_timer->Milliseconds() + preprocessing_timer->Milliseconds() + dedispersion_timer->Milliseconds() + output_timer->Milliseconds();
    runtime_time *= 1e-3; //seconds
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

    // Append the timing results to a log file
    auto total_time = Stopwatch::ToString(gulpEnd.elapsedTime(gulpStart));
    std::ofstream perf_file("perf.log", std::ios::app);
    perf_file << input_timer->ToString() << "\t"
              << output_timer->ToString() << "\t"
              << dedispersion_timer->ToString() << "\t"
              << total_time << std::endl;
    perf_file.close();
#endif
}

} // end namespace dedisp