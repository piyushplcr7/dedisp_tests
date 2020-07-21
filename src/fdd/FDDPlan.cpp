#include "FDDPlan.hpp"

#include <complex>
#include <cmath>
#include <cstring>
#include <iostream>
#include <assert.h>

#include <fftw3.h>
#include <omp.h>
#include <cufft.h>

#include "unpack/unpack.h"
#include "dedisperse/FDDKernel.hpp"

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

void fft_r2c_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    auto plan = fftwf_plan_dft_r2c_1d(n, data, (fftwf_complex *) data, FFTW_ESTIMATE);

    #pragma omp parallel for
    for (unsigned int i = 0; i < batch; i++) {
        auto *in  = &data[i * stride];
        auto *out = (fftwf_complex *) in;
        fftwf_execute_dft_r2c(plan, in, out);
    }

    fftwf_destroy_plan(plan);
}

void fft_c2r_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    auto plan = fftwf_plan_dft_c2r_1d(n, (fftwf_complex *) data, data, FFTW_ESTIMATE);

    #pragma omp parallel for
    for (unsigned int i = 0; i < batch; i++)
    {
        auto *out = &data[i * stride];
        auto *in  = (fftwf_complex *) out;
        fftwf_execute_dft_c2r(plan, in, out);

        for (unsigned int j = 0; j < n; j++)
        {
            double scale = 1.0 / n;
            out[j] *= scale;
        }
    }

    fftwf_destroy_plan(plan);
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
    execute_gpu(nsamps, in, in_nbits, out, out_nbits);
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
    float numax        = m_f0;          // highest frequency
    float dnu          = m_df;          // negative frequency delta
    unsigned int nchan = m_nchans;      // number of observering frequencies
    unsigned int nsamp = nsamps;        // number of time samples
    unsigned int nfreq = (nsamp/2 + 1); // number of spin frequencies
    unsigned int ndm   = m_dm_count;    // number of DMs

    // Compute the number of output samples
    unsigned int nsamp_computed = nsamp - m_max_delay;

    // Debug
    std::cout << "dt    = " << dt << std::endl;
    std::cout << "numax = " << numax << std::endl;
    std::cout << "dnu   = " << dnu << std::endl;
    std::cout << "nchan = " << nchan << std::endl;
    std::cout << "nsamp = " << nsamp << std::endl;
    std::cout << "nfreq = " << nfreq << std::endl;
    std::cout << "ndm   = " << ndm << std::endl;

    // Compute padded number of samples (for r2c transformation)
    unsigned int nsamp_padded = round_up(nsamp + 1, 1024);
    std::cout << "nsamp_padded: " << nsamp_padded << std::endl << std::endl;

    // Allocate memory
    std::vector<float> data_nu;
    std::vector<float> data_dm;
    data_nu.resize((size_t) nchan * nsamp_padded);
    data_dm.resize((size_t) ndm * nsamp_padded);

    // Transpose input and convert to floating point:
    std::cout << "Transpose/convert input" << std::endl;
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
    std::cout << "FFT input r2c" << std::endl;
    fft_r2c_inplace(
        nsamp,           // n
        nchan,           // batch
        nsamp_padded,    // stride
        data_nu.data()); // data

    // Generate spin frequency table
    if (h_spin_frequencies.size() != nfreq)
    {
        generate_spin_frequency_table(nfreq, nsamp, dt);
    }

    // Dedispersion in frequency domain
    std::cout << "Perform dedispersion in frequency domain" << std::endl;
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

    // Fourier transform results back to time domain
    fft_c2r_inplace(
        nsamp,        // n
        ndm,          // batch
        nsamp_padded, // stride
        data_dm.data());

    // Copy output
    std::cout << "Transpose output" << std::endl;
    copy_data<float, float>(
        ndm, nsamp_computed,          // height, width
        nsamp_padded, nsamp_computed, // in stride, out stride
        data_dm.data(),               // input
        (float *) out);
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

    // Compute padded number of samples (for r2c transformation)
    unsigned int nsamp_padded = round_up(nsamp + 1, 1024);

    // Maximum number of DMs computed in one gulp
    unsigned int ndm_gulp_max = 128;
    unsigned int ndm_gulp_current = ndm_gulp_max;

    // Maximum number of channels processed in one gulp
    unsigned int nchan_gulp_max = 256;
    unsigned int nchan_gulp_current = nchan_gulp_max;

    // Compute derived counts
    dedisp_size out_bytes_per_sample = out_nbits / (sizeof(dedisp_byte) *
                                                    BITS_PER_BYTE);
    dedisp_size chans_per_word = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;

    // The number of channel words in the input
    dedisp_size nchan_words = nchan / chans_per_word;

    // The number of channel words proccessed in one gulp
    dedisp_size nchan_words_gulp = nchan_gulp_max / chans_per_word;

    // Allocate device memory
    cu::DeviceMemory d_data_in(nsamp * nchan_words_gulp * sizeof(dedisp_word));
    cu::DeviceMemory d_data_nu(nchan_gulp_max * nsamp_padded * sizeof(float));
    cu::DeviceMemory d_data_dm(ndm_gulp_max * nsamp_padded * sizeof(float));
    cu::HostMemory h_data_dm(ndm * nsamp_padded * sizeof(float));

    // Prepare cuFFT plans
    cufftHandle plan_r2c, plan_c2r;
    cufftResult result;
    int n[] = {(int) nsamp};
    int rnembed[] = {(int) nsamp_padded};   // width in real elements
    int cnembed[] = {(int) nsamp_padded/2}; // width in complex elements
    result = cufftPlanMany(
        &plan_r2c,              // plan
        1, n,                   // rank, n
        rnembed, 1, rnembed[0], // inembed, istride, idist
        cnembed, 1, cnembed[0], // onembed, ostride, odist
        CUFFT_R2C,              // type
        nchan_gulp_max);        // batch
    if (result != CUFFT_SUCCESS)
    {
        throw std::runtime_error("Error creating real to complex FFT plan.");
    }
    result = cufftPlanMany(
        &plan_c2r,              // plan
        1, n,                   // rank, n
        cnembed, 1, cnembed[0], // inembed, istride, idist
        rnembed, 1, rnembed[0], // onembed, ostride, odist
        CUFFT_C2R,              // type
        ndm_gulp_max);          // batch
    if (result != CUFFT_SUCCESS)
    {
        throw std::runtime_error("Error creating complex to real FFT plan.");
    }
    cufftSetStream(plan_r2c, *executestream);
    cufftSetStream(plan_c2r, *executestream);

    // Generate spin frequency table
    if (h_spin_frequencies.size() != nfreq)
    {
        generate_spin_frequency_table(nfreq, nsamp, dt);
        d_spin_frequencies.resize(nfreq * sizeof(dedisp_float));
        htodstream->memcpyHtoDAsync(d_spin_frequencies, h_spin_frequencies.data(), d_spin_frequencies.size());
    }

    // Events
    cu::Event inputCopied, computeFinished, outputCopied;

    //--------------------------------------------------------------------------------
    // Debug
    #if 0
    std::vector<float> data_nu;
    data_nu.resize((size_t) nchan * nsamp_padded);
    transpose_data<const byte_type, float>(
    nchan,           // height
    nsamp,           // width
    nchan,           // in_stride
    nsamp_padded,    // out_stride
    127.5,           // offset
    nchan,           // scale
    in,              // in
    data_nu.data()); // out
    #endif
    //--------------------------------------------------------------------------------

    // Initialize FDDKernel
    FDDKernel kernel;
    kernel.copy_delay_table(
        d_delay_table,
        m_nchans * sizeof(dedisp_float),
        0, *htodstream);

    // Process all channels
    std::cout << "Perform dedispersion in frequency domain" << std::endl;
    for (unsigned ichan_start = 0; ichan_start < nchan; ichan_start += nchan_gulp_current)
    {
        // Compute current number of channels
        nchan_gulp_current = ichan_start + nchan_gulp_current < nchan ?
                             nchan_gulp_current : nchan - ichan_start;
        unsigned int ichan_end = ichan_start + nchan_gulp_current;

        // Info
        std::cout << "Processing channel " << ichan_start << " to " << ichan_end << std::endl;

        // Wait for input buffer to be free
        htodstream->waitEvent(computeFinished);

        // Copy input to device
        dedisp_size dst_stride = nchan_words_gulp * sizeof(dedisp_word);
        dedisp_size src_stride = nchan_words * sizeof(dedisp_word);
        dedisp_size gulp_chan_byte_idx = (ichan_start/chans_per_word) * sizeof(dedisp_word);
        htodstream->memcpyHtoD2DAsync(
            d_data_in,                   // dst
            dst_stride,                  // dst width
            in + gulp_chan_byte_idx,     // src
            src_stride,                  // src width
            dst_stride,                  // width bytes
            nsamp);                      // height
        htodstream->record(inputCopied);

        executestream->waitEvent(inputCopied);
        transpose_unpack(
            d_data_in,        // d_in
            nchan_words_gulp, // input width
            nsamp,            // input height
            nchan_words_gulp, // in_stride
            nsamp_padded,     // out_stride
            d_data_nu,        // d_out
            in_nbits, 32,     // in_nbits, out_nbits
            1.0/nchan,        // scale
            *executestream);  // stream

        //--------------------------------------------------------------------------------
        // Debug
        #if 0
        unsigned int k = 64;
        std::vector<float> temp(nsamp_padded);
        executestream->synchronize();
        for (unsigned int y = 0; y < nchan_gulp_current; y++)
        {
            float *ptr = (float *) d_data_nu;
            ptr += y * nsamp_padded;
            executestream->memcpyDtoHAsync(temp.data(), ptr, nsamp_padded * sizeof(float));
            bool stop = false;
            for (unsigned int x = 0; x < k; x++)
            {
                float tst = temp[x];
                float ref = data_nu[(y+ichan_start) * nsamp_padded + x];
                if ((y == 0 || y == 61) && (ichan_start >= 896))
                {
                    std::clog << "[" << y << "," << x << "]\t"
                              << tst << "\t"
                              << ref << std::endl;
                }
                if (std::abs(ref - tst) > 1e-5)
                {
                    std::clog << "[" << y << "," << x << "]\t"
                              << tst << " != " << ref << std::endl;
                    stop = true;
                }
            }
            if (stop)
            break;
        }
        #endif
        //--------------------------------------------------------------------------------

        // FFT data (real to complex) along time axis
        cufftExecR2C(plan_r2c, d_data_nu, d_data_nu);

        // Process all DMs
        for (unsigned int idm_start = 0; idm_start < ndm; idm_start += ndm_gulp_current)
        {
            // Compute current number of DMs
            ndm_gulp_current = idm_start + ndm_gulp_current < ndm ?
                               ndm_gulp_current : ndm - idm_start;
            unsigned int idm_end = idm_start + ndm_gulp_current;

            // Get pointer to DM output data on host
            dedisp_size dm_stride = nsamp_padded * out_bytes_per_sample;
            dedisp_size dm_offset = idm_start * dm_stride;
            void *h_dm_ptr = (void *) (((size_t) h_data_dm.data()) + dm_offset);

            // Info
            std::cout << "Processing DM " << idm_start << " to " << idm_end << std::endl;

            // Wait for output buffer to be free
            dtohstream->synchronize();

            if (ichan_start == 0)
            {
                // Initialize output to zero
                d_data_dm.zero(*executestream);
            } else {
                // Copy partial result
                htodstream->memcpyHtoDAsync(
                    d_data_dm,                     // dst
                    h_dm_ptr,                      // src
                    ndm_gulp_current * dm_stride); // size
                htodstream->record(inputCopied);
                executestream->waitEvent(inputCopied);
            }

            // Dedispersion in frequency domain
            kernel.launch(
                ndm_gulp_current,   // ndm
                nfreq,              // nfreq
                nchan_gulp_current, // nchan
                dt,                 // dt
                d_spin_frequencies, // d_spin_frequencies
                d_dm_list,          // d_dm_list
                d_data_nu,          // d_in
                d_data_dm,          // d_out
                nsamp_padded/2,     // in stride
                nsamp_padded/2,     // out stride
                idm_start,          // idm_start
                ichan_start,        // ichan_start
                *executestream);    // stream

            if (ichan_end == nchan)
            {
                // Fourier transform results back to time domain
                cufftExecC2R(plan_c2r, d_data_dm, d_data_dm);

                // FFT scaling
                kernel.scale(
                    ndm_gulp_current, // height
                    nsamp_computed,   // width
                    nsamp_padded,     // stride
                    d_data_dm,        // d_data
                    *executestream);  // stream

                // Copy output
                dedisp_size dst_stride = nsamp_computed * out_bytes_per_sample;
                dedisp_size dst_offset = idm_start * dst_stride;
                dedisp_size src_stride = nsamp_padded * out_bytes_per_sample;
                executestream->record(computeFinished);
                dtohstream->waitEvent(computeFinished);
                dtohstream->memcpyDtoH2DAsync(
                    out + dst_offset,  // dst
                    dst_stride,        // dst width
                    d_data_dm,         // src
                    src_stride,        // src width
                    dst_stride,        // width bytes
                    ndm_gulp_current); // height
                dtohstream->record(outputCopied);
            } else {
                // Stash the data for the current DM
                executestream->record(computeFinished);
                dtohstream->waitEvent(computeFinished);
                dtohstream->memcpyDtoHAsync(
                    h_dm_ptr,                      // dst
                    d_data_dm,                     // src
                    ndm_gulp_current * dm_stride); // size
                dtohstream->record(outputCopied);
            }
        } // end for idm_start
    } // end for ichan_start

    // Wait for final memory transfer
    dtohstream->synchronize();
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