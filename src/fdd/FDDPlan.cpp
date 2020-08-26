#include "FDDPlan.hpp"

#include <complex>
#include <cmath>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <thread>

#include <assert.h>
#include <fftw3.h>
#include <omp.h>
#include <cufft.h>

#include "common/dedisp_strings.h"
#include "unpack/unpack.h"
#include "dedisperse/FDDKernel.hpp"
#include "external/Stopwatch.h"

namespace dedisp
{

// Constructor
FDDPlan::FDDPlan(
    size_type  nchans,
    float_type dt,
    float_type f0,
    float_type df) :
    Plan(nchans, dt, f0, df)
{
}

// Destructor
FDDPlan::~FDDPlan()
{}

int round_up(int a, int b)
{
    return ((a + b - 1) / b) * b;
}

// Private helper functions
template<typename InputType, typename OutputType>
void transpose_data(
    size_t height,
    size_t width,
    size_t in_stride,
    size_t out_stride,
    float offset, // use this to undo quantization, e.g. 128 for 8-bit quantization
    float scale,  // use this to prevent overflows when summing the data
    InputType *in,
    OutputType *out)
{
    #pragma omp parallel for
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            auto *src = &in[x * in_stride];
            auto *dst = &out[y * out_stride];
            dst[x] = ((OutputType) src[y] - offset) / scale;
        }
    }
}

template<typename InputType, typename OutputType>
void copy_data(
    size_t height,
    size_t width,
    size_t in_stride,
    size_t out_stride,
    InputType *in,
    OutputType *out)
{
    #pragma omp parallel for
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            auto *src = &in[y * in_stride];
            auto *dst = &out[y * out_stride];
            dst[x] = src[x];
        }
    }
}

void fft_r2c(
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

void fft_r2c_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    fft_r2c(n, batch, stride, stride, data, data);
}

void fft_c2r(
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

void fft_c2r_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    fft_c2r(n, batch, stride, stride, data, data);
}


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
void FDDPlan::execute(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits)
{
    //execute_cpu_segmented(nsamps, in, in_nbits, out, out_nbits);
    execute_cpu(nsamps, in, in_nbits, out, out_nbits);
}

void FDDPlan::execute_cpu(
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
    std::cout << "Initialization time: " << init_timer->ToString() << " sec." << std::endl;
    std::cout << "Preprocessing time:  " << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << "Dedispersion time:   " << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << "Postprocessing time: " << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << "Output memcopy time: " << output_timer->ToString() << " sec." << std::endl;
    std::cout << "Total time:          " << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}

// Chunk metadata
struct Chunk
{
    // Time domain
    unsigned int isamp_start; // scalar index
    unsigned int isamp_end;   // scalar index

    // Frequency domain
    unsigned int ifreq_start; // scalar index
    unsigned int ifreq_end;   // scalar index
    unsigned int nfreq_good;  // number of good samples
};

void compute_chunks(
    unsigned int nsamp,
    unsigned int nsamp_good,
    unsigned int nfft,
    unsigned int nfreq_chunk_padded,
    unsigned int& nfreq_computed,
    std::vector<Chunk>& chunks)
{
    nfreq_computed = 0;

    for (unsigned int ichunk = 0; ichunk < chunks.size(); ichunk++)
    {
        Chunk& chunk = chunks[ichunk];

        // Compute start and end indices in the time domain
        unsigned int isamp_start = ichunk * nsamp_good;
        unsigned int isamp_end   = isamp_start + nfft;
                     isamp_end   = std::min(isamp_end, nsamp);

        // Compute number of scalar samples in the time domain
        unsigned int nsamp = isamp_end - isamp_start;

        // Store time domain parameters
        chunk.isamp_start = isamp_start;
        chunk.isamp_end   = isamp_end;

        // Compute the number of complex samples in the frequency domain
        unsigned int nfreq       = std::ceil(nsamp / 2.0) + 1;
        unsigned int ifreq_start = ichunk * nfreq_chunk_padded;
        unsigned int ifreq_end   = ifreq_start + nfreq;
        chunk.nfreq_good         = nfreq;
        chunk.ifreq_start        = ifreq_start; // complex index
        chunk.ifreq_end          = ifreq_end;   // complex index

        // Number of good spin frequencies computed
        nfreq_computed += ifreq_end - ifreq_start;
    }
}

void print_chunks(
    std::vector<Chunk>& chunks)
{
    for (unsigned int ichunk = 0; ichunk < chunks.size(); ichunk++)
    {
        Chunk& chunk = chunks[ichunk];
        unsigned int nsamp = chunk.isamp_end - chunk.isamp_start;
        printf("Chunk %d: isamp %5d - %5d, nsamp: %5d, ifreq: %5d - %5d, nfreq: %d\n",
            ichunk, chunk.isamp_start, chunk.isamp_end, nsamp,
            chunk.ifreq_start*2, chunk.ifreq_end*2, chunk.nfreq_good*2);
    }
}

void copy_chunk_output(
    float* src,
    float* dst,
    unsigned int ndm,
    unsigned int nsamp,
    unsigned int nsamp_computed,
    unsigned int nsamp_padded,
    unsigned int nsamp_good,
    std::vector<Chunk>& chunks)
{
    size_t ostart = 0;
    for (auto& chunk : chunks)
    {
        size_t istart = chunk.ifreq_start*2;
        size_t oend   = std::min((size_t) nsamp_computed, ostart + nsamp_good);
        size_t nsamp  = oend - ostart;
        if (nsamp > 0)
        {
            auto *src_ptr = &src[istart];
            auto *dst_ptr = &dst[ostart];
            memcpy2D(
                dst_ptr,                        // dstPtr
                nsamp_computed * sizeof(float), // dstWidth
                src_ptr,                        // srcPtr
                nsamp_padded * sizeof(float),   // srcWidth
                nsamp * sizeof(float),          // widthBytes
                ndm);
            ostart += nsamp;
        }
    }
}

void FDDPlan::execute_cpu_segmented(
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
    std::cout << "Initialization time: " << init_timer->ToString() << " sec." << std::endl;
    std::cout << "Preprocessing time:  " << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << "Dedispersion time:   " << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << "Postprocessing time: " << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << "Output memcopy time: " << output_timer->ToString() << " sec." << std::endl;
    std::cout << "Total time:          " << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}

void FDDPlan::execute_gpu(
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
    unsigned int ndm_batch_max = 32;
    unsigned int ndm_fft_batch = 16;
                 ndm_fft_batch = std::min(ndm_batch_max, ndm_fft_batch);
    unsigned int ndm_buffers   = 8;
                 ndm_buffers   = std::min(ndm_buffers, (unsigned int) ((ndm + ndm_batch_max) / ndm_batch_max));

    // Maximum number of channels processed in one gulp
    unsigned int nchan_batch_max = 32;
    unsigned int nchan_fft_batch = 32;
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
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();

    // Allocate memory
    std::cout << memory_alloc_str << std::endl;
    mAllocMem.start();
    cu::HostMemory   h_data_dm(ndm * nsamp_padded * sizeof(float));
    cu::DeviceMemory d_data_nu(nchan_batch_max * nsamp_padded * sizeof(float));
    std::vector<cu::HostMemory> h_data_in_(nchan_buffers);
    std::vector<cu::DeviceMemory> d_data_in_(nchan_buffers);
    std::vector<cu::DeviceMemory> d_data_out_(ndm_buffers);
    for (unsigned int i = 0; i < nchan_buffers; i++)
    {
        h_data_in_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
        d_data_in_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
    }
    for (unsigned int i = 0; i < ndm_buffers; i++)
    {
        d_data_out_[i].resize(ndm_batch_max * nsamp_padded * sizeof(float));
    }
    mAllocMem.end();

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

    // Generate spin frequency table
    mPrepSpinf.start();
    if (h_spin_frequencies.size() != nfreq)
    {
        generate_spin_frequency_table(nfreq, nsamp, dt);
        d_spin_frequencies.resize(nfreq * sizeof(dedisp_float));
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
        job.h_in_ptr      = h_data_in_[job_id % nchan_buffers];
        job.d_in_ptr      = d_data_in_[job_id % nchan_buffers];
        if (job.nchan_current == 0) {
            channel_jobs.pop_back();
        }
    }

    struct DMData{
        unsigned int idm_start;
        unsigned int idm_end;
        unsigned int ndm_current;
        float* h_out_ptr;
        dedisp_float2* d_out_ptr;
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
        job.d_out_ptr   = d_data_out_[job_id % ndm_buffers];
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
                d_data_nu,                           // d_out
                in_nbits, 32,                        // in_nbits, out_nbits
                1.0/nchan,                           // scale
                *executestream);                     // stream

            // Apply zero padding
            auto dst_ptr = ((float *) d_data_nu.data()) + nsamp;
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
                cufftReal    *idata = (cufftReal *) d_data_nu.data() + i * nsamp_padded * nchan_fft_batch;
                cufftComplex *odata = (cufftComplex *) idata;
                cufftExecR2C(plan_r2c, idata, odata);
            }
            executestream->record(channel_job.preprocessingEnd);

            // Initialize output to zero
            if (channel_job_id == 0)
            {
                // Wait for all previous output copies to finish
                dtohstream->synchronize();

                for (cu::DeviceMemory& d_data_out : d_data_out_)
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
                    d_data_nu,                 // d_in
                    dm_job.d_out_ptr,          // d_out
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

            // Add preprocessing time for the current channel job
            preprocessing_timer->Add(channel_job.preprocessingEnd.elapsedTime(channel_job.preprocessingStart));
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
            auto* h_out = (void *) (((size_t) h_data_dm.data()) + dm_offset);
            auto *d_out = (float *) dm_job.d_out_ptr;

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
            dtohstream->memcpyDtoHAsync(
                h_out,                           // dst
                d_out,                           // src
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
    dedisp_size dst_stride = nsamp_computed * out_bytes_per_sample;
    dedisp_size src_stride = nsamp_padded * out_bytes_per_sample;
    memcpy2D(
        out,        // dst
        dst_stride, // dst width
        h_data_dm,  // src
        src_stride, // src width
        dst_stride, // width bytes
        ndm);       // height
    output_timer->Pause();
    mCopyMem.end();
    total_timer->Pause();

    // Accumulate dedispersion and postprocessing time for all dm jobs
    for (auto& job : dm_jobs)
    {
        dedispersion_timer->Add(job.dedispersionEnd.elapsedTime(job.dedispersionStart));
        postprocessing_timer->Add(job.postprocessingEnd.elapsedTime(job.postprocessingStart));
    }

    // Print timings
    std::cout << timings_str << std::endl;
    std::cout << init_time_str           << init_timer->ToString() << " sec." << std::endl;
    std::cout << preprocessing_time_str  << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << dedispersion_time_str   << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << postprocessing_time_str << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << output_memcpy_time_str  << output_timer->ToString() << " sec." << std::endl;
    std::cout << total_time_str          << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}

void FDDPlan::execute_gpu_segmented(
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
        // Repeat spin frequencies several times for the different chunks
        h_spin_frequencies.resize(nsamp_padded);
        #pragma omp parallel for
        for (unsigned int ifreq = 0; ifreq < nfreq_chunk; ifreq++)
        {
            float spin_frequency = ifreq * (1.0/(nfft*dt));
            for (unsigned int ichunk = 0; ichunk < chunks.size(); ichunk++)
            {
                float* dst_ptr = (float *) h_spin_frequencies.data();
                dst_ptr[(1ULL) * ichunk * nfreq_chunk_padded + ifreq] = spin_frequency;
            }
        }

        // Copy segmented spin frequencies to the GPU
        d_spin_frequencies.resize(h_spin_frequencies.size());
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
        float* h_data_t_dm_ptr;
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
                d_data_t_nu,                           // d_out
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

            // Add preprocessing time for the current channel job
            preprocessing_timer->Add(channel_job.preprocessingEnd.elapsedTime(channel_job.preprocessingStart));
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
            auto* d_data_f_dm = (float *) dm_job.d_data_f_dm_ptr;
            auto* d_data_t_dm = (float *) dm_job.d_data_t_dm_ptr;

            // Fourier transform results back to time domain
            executestream->record(dm_job.postprocessingStart);
            for (unsigned int idm = 0; idm < dm_job.ndm_current; idm++)
            {
                auto *idata  = (cufftComplex *) d_data_f_dm + (1ULL * idm * nsamp_padded/2);
                auto *odata = (cufftReal *) d_data_t_dm + (1ULL * idm * nsamp_padded);
                cufftExecC2R(plan_c2r, idata, odata);
            }

            // FFT scaling
            kernel.scale(
                dm_job.ndm_current, // height
                nsamp_padded,       // width
                nsamp_padded,       // stride
                1.0f / nfft,        // scale
                d_data_t_dm,        // d_data
                *executestream);    // stream
            executestream->record(dm_job.postprocessingEnd);

            // Copy output
            dtohstream->waitEvent(dm_job.postprocessingEnd);
            dtohstream->record(dm_job.outputStart);
            dtohstream->memcpyDtoHAsync(
                h_data_t_dm_ptr,                 // dst
                d_data_t_dm,                     // src
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
        dedispersion_timer->Add(job.dedispersionEnd.elapsedTime(job.dedispersionStart));
        postprocessing_timer->Add(job.postprocessingEnd.elapsedTime(job.postprocessingStart));
    }

    // Print timings
    std::cout << timings_str << std::endl;
    std::cout << init_time_str           << init_timer->ToString() << " sec." << std::endl;
    std::cout << preprocessing_time_str  << preprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << dedispersion_time_str   << dedispersion_timer->ToString() << " sec." << std::endl;
    std::cout << postprocessing_time_str << postprocessing_timer->ToString() << " sec." << std::endl;
    std::cout << output_memcpy_time_str  << output_timer->ToString() << " sec." << std::endl;
    std::cout << total_time_str          << total_timer->ToString() << " sec." << std::endl;
    std::cout << std::endl;
}

// Private helper functions
void FDDPlan::generate_spin_frequency_table(
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

} // end namespace dedisp