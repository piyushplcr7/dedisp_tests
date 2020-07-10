#include "FDDPlan.hpp"

#include <complex>
#include <cmath>
#include <cstring>
#include <iostream>

#include <fftw3.h>
#include <omp.h>

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

void FDDPlan::execute(
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
    std::vector<float> t_nu;
    std::vector<float> f_dm;
    t_nu.resize((size_t) nchan * nsamp_padded);
    f_dm.resize((size_t) ndm * nsamp_padded);

    // Transpose input and convert to floating point:
    std::cout << "Transpose/convert input" << std::endl;
    #pragma omp parallel for
    for (unsigned int ichan = 0; ichan < nchan; ichan++) {
        for (unsigned int isamp = 0; isamp < nsamp; isamp++) {
            const byte_type *ptr = in + (isamp * nchan);
            t_nu[ichan * nsamp_padded + isamp] = ((float) ptr[ichan]) - 127.5f;
        }
    }

    // FFT data (real to complex) along time axis
    std::cout << "FFT input r2c" << std::endl;
    fftwf_plan plan_r2c = fftwf_plan_dft_r2c_1d(nsamp, t_nu.data(), (fftwf_complex *) t_nu.data(), FFTW_ESTIMATE);
    #pragma omp parallel for
    for (unsigned int ichan = 0; ichan < nchan; ichan++) {
        float *in = &t_nu[ichan * nsamp_padded];
        fftwf_complex *out = (fftwf_complex *) in;
        fftwf_execute_dft_r2c(plan_r2c, in, out);
    }

    // DM delays
    float tdms[ndm][nchan];
    for (unsigned int ichan = 0; ichan < nchan; ichan++) {
        for (unsigned int idm = 0; idm < ndm; idm++) {
            float dm = h_dm_list[idm];
            tdms[idm][ichan] = dm * h_delay_table[ichan] * dt;
        }
    }

    std::cout << "Perform dedispersion in frequency domain" << std::endl;

    // Loop over DM
    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        // Loop over spin frequencies
        for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
        {
            // Compute initial phasor value
            float f = ifreq * dt;
            float phase0 = (2.0 * M_PI * f * tdms[idm][0]);
            std::complex<float> phasor(cosf(phase0), sinf(phase0));

            // Compute delta phasor
            float phase1 = (2.0 * M_PI * f * tdms[idm][1]);
            float phase_delta = phase1 - phase0;
            std::complex<float> phasor_delta(cosf(phase_delta), sinf(phase_delta));

            // Sum over observing frequencies
            std::complex<float> sum = {0, 0};

            // Loop over observing frequencies
            for (unsigned int ichan = 0; ichan < nchan; ichan++)
            {
                // Complex multiply and add
                std::complex<float>* sample = (std::complex<float> *) &t_nu[ichan * nsamp_padded + ifreq];
                sum += *sample * phasor;

                // Update phasor
                phasor *= phasor_delta;
            }

            // Store sum
            std::complex<float>* dst_ptr = (std::complex<float> *) &f_dm[idm * nsamp_padded + ifreq];
            *dst_ptr = sum;
        }
    }

    // Fourier transform results back to time domain
    std::cout << "FFT output c2r" << std::endl;
    fftwf_plan plan_c2r = fftwf_plan_dft_c2r_1d(nsamp, (fftwf_complex *) f_dm.data(), f_dm.data(), FFTW_ESTIMATE);
    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        float *out        = &f_dm[idm * nsamp_padded];
        fftwf_complex *in = (fftwf_complex *) out;
        fftwf_execute_dft_c2r(plan_c2r, in, out);

        for (unsigned int isamp = 0; isamp < nsamp; isamp++)
        {
            float fft_scale = 1.0f / nsamp;
            float chan_scale = 1.0f / nchan;
            out[isamp] *= fft_scale * chan_scale;
        }
    }

    // Copy output
    std::cout << "Transpose output" << std::endl;
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        for (unsigned int isamp = 0; isamp < nsamp_computed; isamp++)
        {
            float *src_ptr = &f_dm[idm * nsamp_padded];
            float *dst_ptr = ((float *) out) + (idm * nsamp_computed);
            dst_ptr[isamp] = std::abs(src_ptr[isamp]);
        }
    }

    // Free memory
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
}

} // end namespace dedisp