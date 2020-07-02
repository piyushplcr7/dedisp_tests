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

    // Sizes
    size_t t_nu_bytes = nchan * nsamp_padded * sizeof(float);
    size_t f_dm_bytes = ndm * nsamp_padded * sizeof(float);

    // Allocate memory
    float *t_nu = (float *) fftwf_malloc(t_nu_bytes);
    float *f_dm = (float *) fftwf_malloc(f_dm_bytes);

    // Reset memory to zero
    std::memset((void *) t_nu, 0, t_nu_bytes);
    std::memset((void *) f_dm, 0, f_dm_bytes);

    // Transpose input and convert to floating point:
    std::cout << "Transpose/convert input" << std::endl;
    #pragma omp parallel for
    for (unsigned int ichan = 0; ichan < nchan; ichan++) {
        for (unsigned int isamp = 0; isamp < nsamp; isamp++) {
            const byte_type *src_ptr = in + (isamp * nchan);
            float *dst_ptr = t_nu + (ichan * nsamp_padded);
            dst_ptr[isamp] = ((float) src_ptr[ichan]) - 127.5f;
        }
    }

    // FFT data (real to complex) along time axis
    std::cout << "FFT input r2c" << std::endl;
    fftwf_plan plan_r2c = fftwf_plan_dft_r2c_1d(nsamp, t_nu, (fftwf_complex *) t_nu, FFTW_ESTIMATE);
    #pragma omp parallel for
    for (unsigned int ichan = 0; ichan < nchan; ichan++) {
        float *in          = t_nu + (ichan * nsamp_padded);
        fftwf_complex *out = (fftwf_complex *) in;
        fftwf_execute_dft_r2c(plan_r2c, in, out);
    }

    // Compute spin frequencies (FFT'ed axis of time)
    float f[nfreq];
    for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++) {
        f[ifreq] = ifreq * dt * 1e6;
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
        // Loop over observing frequencies
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            // Loop over spin frequencies
            for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
            {
                // Compute phase
                float phase = 2.0 * M_PI * f[ifreq] * tdms[idm][ichan];

                // Compute phasor
                std::complex<float> phasor(cosf(phase), sinf(phase));

                // Complex multiply and add
                fftwf_complex *src_ptr = (fftwf_complex *) (t_nu + ichan * nsamp_padded);
                fftwf_complex *dst_ptr = (fftwf_complex *) (f_dm + idm * nsamp_padded);
                float real = src_ptr[ifreq][0];
                float imag = src_ptr[ifreq][1];
                dst_ptr[ifreq][0] += real * phasor.real() - imag * phasor.imag();
                dst_ptr[ifreq][1] += real * phasor.imag() + imag * phasor.real();
            }
        }
    }

    // Fourier transform results back to time domain
    std::cout << "FFT output c2r" << std::endl;
    fftwf_plan plan_c2r = fftwf_plan_dft_c2r_1d(nsamp, (fftwf_complex *) f_dm, f_dm, FFTW_ESTIMATE);
    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        float *out        = f_dm + (idm * nsamp_padded);
        fftwf_complex *in = (fftwf_complex *) out;
        fftwf_execute_dft_c2r(plan_c2r, in, out);

        for (unsigned int isamp = 0; isamp < nsamp; isamp++)
        {
            float scale = 1.0f / nsamp;
            out[isamp] *= scale;
        }
    }

    // Copy output
    std::cout << "Transpose output" << std::endl;
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        for (unsigned int isamp = 0; isamp < nsamp_computed; isamp++)
        {
            float *src_ptr = f_dm + (idm * nsamp_padded);
            float *dst_ptr = ((float *) out) + (idm * nsamp_computed);
            dst_ptr[isamp] = std::abs(src_ptr[isamp]);
        }
    }

    // Free memory
    free(t_nu);
    free(f_dm);
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
}

} // end namespace dedisp