#include <FDDCPUPlan.hpp>

#include <complex>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>

#include <fftw3.h>
#include <omp.h>

#include "common/dedisp_strings.h"
#include "external/Stopwatch.h"

#include "common/helper.h"
#include "helper.h"
#include "chunk.h"

namespace dedisp
{

// Constructor
FDDCPUPlan::FDDCPUPlan(
    size_type  nchans,
    float_type dt,
    float_type f0,
    float_type df) :
    Plan(nchans, dt, f0, df)
{
}

// Destructor
FDDCPUPlan::~FDDCPUPlan()
{}

// Dedisperison kernels
template<typename InputType, typename OutputType>
void dedisperse_reference(
    unsigned int ndm,
    unsigned int nfreq,
    unsigned int nchan,
    float dt,      // sample time
    float *f,      // spin frequencies
    float *dms,    // DMs
    float *delays, // delay table
    size_t in_stride,
    size_t out_stride,
    std::complex<InputType> *in,
    std::complex<OutputType> *out)
{
    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        // DM delays
        float tdms[nchan];
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            tdms[ichan] = dms[idm] * delays[ichan] * dt;
        }

        // Loop over spin frequencies
        for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
        {
            // Sum over observing frequencies
            std::complex<OutputType> sum = {0, 0};

            // Loop over observing frequencies
            for (unsigned int ichan = 0; ichan < nchan; ichan++)
            {
                // Compute phasor
                float phase = (2.0 * M_PI * f[ifreq] * tdms[ichan]);
                std::complex<float> phasor(cosf(phase), sinf(phase));

                // Load sample
                auto* sample_ptr = &in[ichan * in_stride];
                auto  sample     = (std::complex<float>) sample_ptr[ifreq];

                // Update sum
                sum += sample * phasor;
            }

            // Store sum
            auto* dst_ptr = &out[idm * out_stride];
            dst_ptr[ifreq] = sum;
        }
    }
}

template<typename InputType, typename OutputType>
void dedisperse_segmented_reference(
    unsigned int ndm,
    unsigned int nchan,
    float dt,      // sample time
    float *f,      // spin frequencies
    float *dms,    // DMs
    float *delays, // delay table
    size_t in_stride,
    size_t out_stride,
    unsigned int nchunk,
    unsigned int nfreq_chunk,
    unsigned int nfreq_chunk_padded,
    std::complex<InputType> *in,
    std::complex<OutputType> *out)
{
    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        // DM delays
        float tdms[nchan];
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            tdms[ichan] = dms[idm] * delays[ichan] * dt;
        }

        // Loop over chunks
        for (unsigned int ichunk = 0; ichunk < nchunk; ichunk++)
        {
            // Loop over spin frequencies
            for (unsigned int ifreq = 0; ifreq < nfreq_chunk; ifreq++)
            {
                // Sum over observing frequencies
                std::complex<OutputType> sum = {0, 0};

                // Loop over observing frequencies
                for (unsigned int ichan = 0; ichan < nchan; ichan++)
                {
                    // Compute phasor
                    float phase = (2.0 * M_PI * f[ifreq] * tdms[ichan]);
                    std::complex<float> phasor(cosf(phase), sinf(phase));

                    // Load sample
                    auto* sample_ptr = &in[ichan * in_stride + ichunk * nfreq_chunk_padded];
                    auto  sample     = (std::complex<float>) sample_ptr[ifreq];

                    // Update sum
                    sum += sample * phasor;
                }

                // Store sum
                auto* dst_ptr = &out[idm * out_stride + ichunk * nfreq_chunk_padded];
                dst_ptr[ifreq] = sum;
            } // end for ifreq
        } // end for chunk
    } // end for idm
}

template<typename InputType, typename OutputType>
void dedisperse_segmented_optimized(
    unsigned int ndm,
    unsigned int nchan,
    float dt,      // sample time
    float *f,      // spin frequency
    float *dms,    // DMs
    float *delays, // delay table
    size_t in_stride,
    size_t out_stride,
    unsigned int nchunk,
    unsigned int nfreq_chunk,
    unsigned int nfreq_chunk_padded,
    std::complex<InputType> *in,
    std::complex<OutputType> *out)
{
    // Transpose input data
    float in_real[nchunk * nfreq_chunk_padded][nchan];
    float in_imag[nchunk * nfreq_chunk_padded][nchan];
    #pragma omp parallel for
    for (unsigned int ifreq = 0; ifreq < nchunk * nfreq_chunk_padded; ifreq++)
    {
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            auto* sample = &in[ichan * in_stride];
            in_real[ifreq][ichan] = sample[ifreq].real();
            in_imag[ifreq][ichan] = sample[ifreq].imag();
        }
    }

    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        // DM delays
        float tdms[nchan];
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            tdms[ichan] = dms[idm] * delays[ichan] * dt;
        }

        // Loop over chunks
        for (unsigned int ichunk = 0; ichunk < nchunk; ichunk++)
        {
            // Loop over spin frequencies
            for (unsigned int ifreq = 0; ifreq < nfreq_chunk; ifreq++)
            {
                unsigned int nchan_batch = 32;

                // Partial sums
                float sums_real[nchan_batch];
                float sums_imag[nchan_batch];
                memset(sums_real, 0, nchan_batch * sizeof(float));
                memset(sums_imag, 0, nchan_batch * sizeof(float));

                for (unsigned int ichan_outer = 0; ichan_outer < nchan; ichan_outer += nchan_batch)
                {
                    float phasors_real[nchan_batch];
                    float phasors_imag[nchan_batch];

                    // Compute phases
                    float phases[nchan_batch];
                    for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                    {
                        phases[ichan_inner] = (2.0 * M_PI * f[ifreq] * tdms[ichan_outer + ichan_inner]);
                    }

                    // Compute phasors
                    for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                    {
                        phasors_real[ichan_inner] = cosf(phases[ichan_inner]);
                        phasors_imag[ichan_inner] = sinf(phases[ichan_inner]);
                    }

                    // Loop over observing frequencies
                    for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                    {
                        unsigned int ichan = ichan_outer + ichan_inner;

                        // Load sample
                        float sample_real = in_real[ichunk * nfreq_chunk_padded + ifreq][ichan];
                        float sample_imag = in_imag[ichunk * nfreq_chunk_padded + ifreq][ichan];

                        // Update sum
                        sums_real[ichan_inner] += sample_real * phasors_real[ichan_inner];
                        sums_real[ichan_inner] -= sample_imag * phasors_imag[ichan_inner];
                        sums_imag[ichan_inner] += sample_real * phasors_imag[ichan_inner];
                        sums_imag[ichan_inner] += sample_imag * phasors_real[ichan_inner];
                    }
                }

                // Sum over observing frequencies
                OutputType sum_real = 0;
                OutputType sum_imag = 0;
                for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                {
                    sum_real += sums_real[ichan_inner];
                    sum_imag += sums_imag[ichan_inner];
                }

                // Store sum
                auto* dst_ptr = &out[idm * out_stride];
                dst_ptr[ichunk * nfreq_chunk_padded + ifreq] = {sum_real, sum_imag};
            }
        }
    }
}

template<typename InputType, typename OutputType, unsigned int nchan_batch, bool extrapolate>
void dedisperse_optimized(
    unsigned int ndm,
    unsigned int nfreq,
    unsigned int nchan,
    float dt,      // sample time
    float *f,      // spin frequencies
    float *dms,    // DMs
    float *delays, // delay table
    size_t in_stride,
    size_t out_stride,
    std::complex<InputType> *in,
    std::complex<OutputType> *out)
{
    // Transpose input data
    float in_real[nfreq][nchan];
    float in_imag[nfreq][nchan];
    #pragma omp parallel for
    for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
    {
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            auto* sample = &in[ichan * in_stride];
            in_real[ifreq][ichan] = sample[ifreq].real();
            in_imag[ifreq][ichan] = sample[ifreq].imag();
        }
    }

    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        // DM delays
        float tdms[nchan];
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            tdms[ichan] = dms[idm] * delays[ichan] * dt;
        }

        // Loop over spin frequencies
        for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
        {
            // Partial sums
            float sums_real[nchan_batch];
            float sums_imag[nchan_batch];
            memset(sums_real, 0, nchan_batch * sizeof(float));
            memset(sums_imag, 0, nchan_batch * sizeof(float));

            for (unsigned int ichan_outer = 0; ichan_outer < nchan; ichan_outer += nchan_batch)
            {
                float phasors_real[nchan_batch];
                float phasors_imag[nchan_batch];

                if (extrapolate)
                {
                    // Compute initial phasor value
                    float phase0 = (2.0 * M_PI * f[ifreq] * tdms[ichan_outer]);
                    float phasor_real = cosf(phase0);
                    float phasor_imag = sinf(phase0);

                    // Compute delta phasor
                    float phase1 = (2.0 * M_PI * f[ifreq] * tdms[ichan_outer + 1]);
                    float phase_delta = phase1 - phase0;
                    float phasor_delta_real = cosf(phase_delta);
                    float phasor_delta_imag = sinf(phase_delta);

                    // Compute phasors
                    for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                    {
                        phasors_real[ichan_inner] = phasor_real;
                        phasors_imag[ichan_inner] = phasor_imag;
                        float temp_real = phasor_real;
                        float temp_imag = phasor_imag;
                        phasor_real = temp_real * phasor_delta_real
                                    - temp_imag * phasor_delta_imag;
                        phasor_imag = temp_real * phasor_delta_imag
                                    + temp_imag * phasor_delta_real;
                    }
                } else {
                    // Compute phases
                    float phases[nchan_batch];
                    for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                    {
                        phases[ichan_inner] = (2.0 * M_PI * f[ifreq] * tdms[ichan_outer]); // + ichan_inner?
                    }

                    // Compute phasors
                    for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                    {
                        phasors_real[ichan_inner] = cosf(phases[ichan_inner]);
                        phasors_imag[ichan_inner] = sinf(phases[ichan_inner]);
                    }
                }

                // Loop over observing frequencies
                for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                {
                    unsigned int ichan = ichan_outer + ichan_inner;

                    // Load sample
                    float sample_real = in_real[ifreq][ichan];
                    float sample_imag = in_imag[ifreq][ichan];

                    // Update sum
                    sums_real[ichan_inner] += sample_real * phasors_real[ichan_inner];
                    sums_real[ichan_inner] -= sample_imag * phasors_imag[ichan_inner];
                    sums_imag[ichan_inner] += sample_real * phasors_imag[ichan_inner];
                    sums_imag[ichan_inner] += sample_imag * phasors_real[ichan_inner];
                }
            }

            // Sum over observing frequencies
            OutputType sum_real = 0;
            OutputType sum_imag = 0;
            for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
            {
                sum_real += sums_real[ichan_inner];
                sum_imag += sums_imag[ichan_inner];
            }

            // Store sum
            auto* dst_ptr = &out[idm * out_stride];
            dst_ptr[ifreq] = {sum_real, sum_imag};
        }
    }
}

// Public interface
void FDDCPUPlan::execute(
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
        execute_cpu_segmented(nsamps, in, in_nbits, out, out_nbits);
    } else {
        execute_cpu(nsamps, in, in_nbits, out, out_nbits);
    }
}


// Private interface
void FDDCPUPlan::execute_cpu(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits)
{
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

    // Timers
    std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();

    // Allocate memory
    std::cout << memory_alloc_str << std::endl;
    std::vector<float> data_nu;
    std::vector<float> data_dm;
    data_nu.resize((size_t) nchan * nsamp_padded);
    data_dm.resize((size_t) ndm * nsamp_padded);

    // Generate spin frequency table
    if (h_spin_frequencies.size() != nfreq)
    {
        generate_spin_frequency_table(nfreq, nsamp, dt);
    }
    init_timer->Pause();

    // Transpose input and convert to floating point
    std::cout << prepare_input_str << std::endl;
    preprocessing_timer->Start();
    transpose_data<const byte_type, float>(
        nchan,           // height
        nsamp,           // width
        nchan,           // in_stride
        nsamp_padded,    // out_stride
        127.5,           // offset
        nchan,           // scale
        in,              // in
        data_nu.data()); // out

    // FFT data (real to complex) along time axis
    std::cout << fft_r2c_str << std::endl;
    fft_r2c_inplace(
        nsamp_fft,       // n
        nchan,           // batch
        nsamp_padded,    // stride
        data_nu.data()); // data
    preprocessing_timer->Pause();

    // Dedispersion in frequency domain
    std::cout << fdd_dedispersion_str << std::endl;

    char* use_reference_str = getenv("USE_REFERENCE");
    bool use_reference = !use_reference_str ? false : atoi(use_reference_str);
    if (use_reference)
    {
        std::cout << "Running reference implementation" << std::endl;
        dedispersion_timer->Start();
        dedisperse_reference<float, float>(
            ndm, nfreq, nchan,                      // data dimensions
            dt,                                     // sample time
            h_spin_frequencies.data(),              // spin frequencies
            h_dm_list.data(),                       // DMs
            h_delay_table.data(),                   // delay table
            nsamp_padded/2,                         // in stride
            nsamp_padded/2,                         // out stride
            (std::complex<float> *) data_nu.data(), // input
            (std::complex<float> *) data_dm.data()  // output
        );
    }
    else
    {
        std::cout << "Running optimized implementation" << std::endl;
        dedispersion_timer->Start();
        dedisperse_optimized<float, float, 32, true>(
            ndm, nfreq, nchan,                      // data dimensions
            dt,                                     // sample time
            h_spin_frequencies.data(),              // spin frequencies
            h_dm_list.data(),                       // DMs
            h_delay_table.data(),                   // delay table
            nsamp_padded/2,                         // in stride
            nsamp_padded/2,                         // out stride
            (std::complex<float> *) data_nu.data(), // input
            (std::complex<float> *) data_dm.data()  // output
        );
    }
    dedispersion_timer->Pause();

    // Fourier transform results back to time domain
    std::cout << fft_c2r_str << std::endl;
    postprocessing_timer->Start();
    fft_c2r_inplace(
        nsamp_fft,    // n
        ndm,          // batch
        nsamp_padded, // stride
        data_dm.data());
    postprocessing_timer->Pause();

    // Copy output
    std::cout << copy_output_str << std::endl;
    output_timer->Start();
    copy_data<float, float>(
        ndm, nsamp_computed,          // height, width
        nsamp_padded, nsamp_computed, // in stride, out stride
        data_dm.data(),               // input
        (float *) out);
    output_timer->Pause();
    total_timer->Pause();

    // Print timings
    std::cout << timings_str << std::endl;
    std::cout << init_time_str              << init_timer->ToString() << " sec." << std::endl;
    std::cout << preprocessing_time_str     << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << dedispersion_time_str      << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << postprocessing_time_str    << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << output_memcpy_time_str     << output_timer->ToString() << " sec." << std::endl;
    std::cout << total_time_str             << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}


void FDDCPUPlan::execute_cpu_segmented(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits)
{
    // Parameters
    float dt           = m_dt;       // sample time
    unsigned int nchan = m_nchans;   // number of observering frequencies
    unsigned int nsamp = nsamps;     // number of time samples
    unsigned int ndm   = m_dm_count; // number of DMs
    unsigned int nfft  = 16384;      // number of samples processed in a segment

    // Compute the number of output samples
    unsigned int nsamp_computed = nsamp - m_max_delay;

    // Compute the number of chunks
    unsigned int nsamp_dm   = std::ceil(m_max_delay);
    while (nfft < nsamp_dm) { nfft *= 2; };
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

    // Timers
    std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();

    // Allocate memory
    std::cout << memory_alloc_str << std::endl;
    std::vector<float> data_t_nu((size_t) nchan * nsamp_padded);
    std::vector<std::complex<float>> data_f_nu((size_t) nchan * nsamp_padded/2);

    // Generate spin frequency table
    if (h_spin_frequencies.size() != nfreq_chunk)
    {
        generate_spin_frequency_table(nfreq_chunk, nfft, dt);
    }
    init_timer->Pause();

    // Transpose input and convert to floating point:
    std::cout << prepare_input_str << std::endl;
    preprocessing_timer->Start();
    transpose_data<const byte_type, float>(
        nchan,             // height
        nsamp,             // width
        nchan,             // in_stride
        nsamp_padded,      // out_stride
        127.5,             // offset
        nchan,             // scale
        in,                // in
        data_t_nu.data()); // out

    // Compute chunks
    std::vector<Chunk> chunks(nchunk);
    unsigned int nfreq_computed;
    compute_chunks(
        nsamp, nsamp_good, nfft,
        nfreq_chunk_padded, nfreq_computed, chunks);

    // Debug
    std::cout << debug_str << std::endl;
    print_chunks(chunks);
    std::cout << "nfreq_computed = " << nfreq_computed << std::endl;
    std::cout << "nsamp_computed = " << nsamp_computed << std::endl;

    // FFT data (real to complex) along time axis
    std::cout << fft_r2c_str << std::endl;
    const int n[] = {(int) nfft};
    int inembed_r2c[] = {(int) nsamp_good};
    int onembed_r2c[] = {(int) nfreq_chunk_padded};
    auto plan_r2c = fftwf_plan_many_dft_r2c(
        1, n,                               // rank, n
        nchunk,                             // howmany
        data_t_nu.data(),                   // in
        inembed_r2c, 1, inembed_r2c[0],     // inembed, istride, idist
        (fftwf_complex *) data_f_nu.data(), // out
        onembed_r2c, 1, onembed_r2c[0],     // onembed, ostride, odist
        FFTW_ESTIMATE);                     // flags
    #pragma omp parallel for
    for (unsigned int ichan = 0; ichan < nchan; ichan++)
    {
        auto *in  = (float *) data_t_nu.data() + (1ULL * ichan * nsamp_padded);
        auto *out = (fftwf_complex *) data_f_nu.data() + (1ULL * ichan * nsamp_padded/2);
        fftwf_execute_dft_r2c(plan_r2c, in, out);
    }
    fftwf_destroy_plan(plan_r2c);
    preprocessing_timer->Pause();

    // Free input buffer
    data_t_nu.resize(0);

    // Allocate buffers
    std::cout << memory_alloc_str << std::endl;
    init_timer->Start();
    std::vector<std::complex<float>> data_f_dm((size_t) ndm * nsamp_padded/2);
    std::vector<float> data_t_dm((size_t) ndm * nsamp_padded);
    init_timer->Pause();

    // Perform dedispersion
    std::cout << fdd_dedispersion_str << std::endl;
    char* use_reference_str = getenv("USE_REFERENCE");
    bool use_reference = !use_reference_str ? false : atoi(use_reference_str);
    if (use_reference)
    {
        std::cout << "Running reference implementation" << std::endl;
        dedispersion_timer->Start();
        dedisperse_segmented_reference<float, float>(
            ndm, nchan,                              // data dimensions
            dt,                                      // sample time
            h_spin_frequencies.data(),               // spin frequencies
            h_dm_list.data(),                        // DMs
            h_delay_table.data(),                    // delay table
            nsamp_padded/2,                          // in stride
            nsamp_padded/2,                          // out stride
            nchunk, nfreq_chunk, nfreq_chunk_padded, // chunk parameters
            data_f_nu.data(),                        // input
            data_f_dm.data()                         // output
        );
    }
    else
    {
        std::cout << "Running optimized implementation" << std::endl;
        dedispersion_timer->Start();
        dedisperse_segmented_optimized<float, float>(
            ndm, nchan,                              // data dimensions
            dt,                                      // sample time
            h_spin_frequencies.data(),               // spin frequencies
            h_dm_list.data(),                        // DMs
            h_delay_table.data(),                    // delay table
            nsamp_padded/2,                          // in stride
            nsamp_padded/2,                          // out stride
            nchunk, nfreq_chunk, nfreq_chunk_padded, // chunk parameters
            data_f_nu.data(),                        // input
            data_f_dm.data()                         // output
        );
    }
    dedispersion_timer->Pause();

    // Fourier transform results back to time domain
    std::cout << fft_c2r_str << std::endl;
    postprocessing_timer->Start();
    int inembed_c2r[] = {(int) nfreq_chunk_padded};
    int onembed_c2r[] = {(int) nfreq_chunk_padded*2};
    auto plan_c2r = fftwf_plan_many_dft_c2r(
        1, n,                               // rank, n
        nchunk,                             // howmany
        (fftwf_complex *) data_f_dm.data(), // in
        inembed_c2r, 1, inembed_c2r[0],     // inembed, istride, idist
        data_t_dm.data(),                   // out
        onembed_c2r, 1, onembed_c2r[0],     // onembed, ostride, odist
        FFTW_ESTIMATE);                     // flags
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        auto *in  = (fftwf_complex *) data_f_dm.data() + (1ULL * idm * nsamp_padded/2);
        auto *out = (float *) data_t_dm.data() + (1ULL * idm * nsamp_padded);
        fftwf_execute_dft_c2r(plan_c2r, in, out);
        for (unsigned int j = 0; j < nsamp_padded; j++)
        {
            double scale = 1.0 / nfft;
            out[j] *= scale;
        }
    }
    fftwf_destroy_plan(plan_c2r);
    postprocessing_timer->Pause();

    // Copy output
    std::cout << copy_output_str << std::endl;
    output_timer->Start();
    copy_chunk_output(
        data_t_dm.data(), (float *) out,
        ndm, nsamp, nsamp_computed,
        nsamp_padded, nsamp_good, chunks);
    output_timer->Pause();
    total_timer->Pause();

    // Print timings
    std::cout << timings_str << std::endl;
    std::cout << init_time_str              << init_timer->ToString() << " sec." << std::endl;
    std::cout << preprocessing_time_str     << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << dedispersion_time_str      << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << postprocessing_time_str    << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << output_memcpy_time_str     << output_timer->ToString() << " sec." << std::endl;
    std::cout << total_time_str             << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}

// Private helper functions
void FDDCPUPlan::generate_spin_frequency_table(
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
}

void FDDCPUPlan::fft_r2c(
    unsigned int n,
    unsigned int batch,
    size_t in_stride,
    size_t out_stride,
    float *in,
    float *out)
{
    auto plan = fftwf_plan_dft_r2c_1d(n, in, (fftwf_complex *) in, FFTW_ESTIMATE);

    #pragma omp parallel for
    for (unsigned int i = 0; i < batch; i++) {
        auto *in_ptr  = &in[i * in_stride];
        auto *out_ptr = (fftwf_complex *) &out[i * out_stride];;
        fftwf_execute_dft_r2c(plan, in_ptr, out_ptr);
    }

    fftwf_destroy_plan(plan);
}

void FDDCPUPlan::fft_r2c_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    fft_r2c(n, batch, stride, stride, data, data);
}

void FDDCPUPlan::fft_c2r(
    unsigned int n,
    unsigned int batch,
    size_t in_stride,
    size_t out_stride,
    float *in,
    float *out)
{
    auto plan = fftwf_plan_dft_c2r_1d(n, (fftwf_complex *) in, out, FFTW_ESTIMATE);

    #pragma omp parallel for
    for (unsigned int i = 0; i < batch; i++)
    {
        auto *in_ptr  = (fftwf_complex *) &in[i * in_stride];
        auto *out_ptr = &out[i * out_stride];
        fftwf_execute_dft_c2r(plan, in_ptr, out_ptr);

        for (unsigned int j = 0; j < n; j++)
        {
            double scale = 1.0 / n;
            out_ptr[j] *= scale;
        }
    }

    fftwf_destroy_plan(plan);
}

void FDDCPUPlan::fft_c2r_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    fft_c2r(n, batch, stride, stride, data, data);
}

} // end namespace dedisp