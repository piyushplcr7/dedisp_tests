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

    // Sizes
    size_t i_t_nu_bytes = nchan * nsamp * sizeof(float);
    size_t z_f_nu_bytes = nchan * nfreq * sizeof(fftwf_complex);
    size_t z_f_dm_bytes = ndm * nfreq * sizeof(fftwf_complex);
    size_t i_t_dm_bytes = ndm * nsamp * sizeof(float);

    // Allocate memory
    float *i_t_nu = (float *) fftwf_malloc(i_t_nu_bytes);
    float *i_t_dm = (float *) fftwf_malloc(i_t_dm_bytes);
    fftwf_complex *z_f_nu = (fftwf_complex *) fftwf_malloc(z_f_nu_bytes);
    fftwf_complex *z_f_dm = (fftwf_complex *) fftwf_malloc(z_f_dm_bytes);

    // Reset memory to zero
    std::memset((void *) i_t_nu, 0, i_t_nu_bytes);
    std::memset((void *) z_f_nu, 0, z_f_nu_bytes);
    std::memset((void *) z_f_dm, 0, z_f_dm_bytes);
    std::memset((void *) i_t_dm, 0, i_t_dm_bytes);

    // Transpose input and convert to floating point:
    std::cout << "Transpose/convert input" << std::endl;
    #pragma omp parallel for
    for (unsigned int isamp = 0; isamp < nsamp; isamp++)
    {
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            const byte_type *src_ptr = in + isamp * nchan;
            float *dst_ptr = (float *) i_t_nu + ichan * nsamp;
            dst_ptr[ichan] = ((float) src_ptr[ichan]) - 127.5f;
        }
    }

    // FFT data (real to complex) along time axis
    std::cout << "FFT input r2c" << std::endl;
    fftwf_plan plan_r2c = fftwf_plan_dft_r2c_1d(nsamp, i_t_nu, z_f_nu, FFTW_ESTIMATE);
    #pragma omp parallel for
    for (unsigned int ichan = 0; ichan < nchan; ichan++) {
        float *in          = i_t_nu + ichan * nsamp;
        fftwf_complex *out = z_f_nu + ichan * nfreq;
        fftwf_execute_dft_r2c(plan_r2c, in, out);
    }

    // Compute spin frequencies (FFT'ed axis of time)
    float f[nfreq];
    for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++) {
        f[ifreq] = roundf(ifreq * dt * 1e6);
    }

    // DM delays
    float tdms[ndm][nchan];
    for (unsigned int ichan = 0; ichan < nchan; ichan++) {
        for (unsigned int idm = 0; idm < ndm; idm++) {
            float dm = h_dm_list[idm];
            tdms[idm][ichan] = dm * h_delay_table[ichan];
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
                fftwf_complex *src_ptr = z_f_nu + ichan * nfreq;
                float real = src_ptr[ifreq][0];
                float imag = src_ptr[ifreq][1];
                fftwf_complex *dst_ptr = z_f_dm + idm * nfreq;
                dst_ptr[ifreq][0] += real * phasor.real() - imag * phasor.imag();
                dst_ptr[ifreq][1] += real * phasor.imag() + imag * phasor.real();
            }
        }
    }

    // Fourier transform results back to time domain
    std::cout << "FFT output c2r" << std::endl;
    fftwf_plan plan_c2r = fftwf_plan_dft_c2r_1d(nsamp, z_f_dm, i_t_dm, FFTW_ESTIMATE);
    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        fftwf_complex *in = z_f_dm + idm * nfreq;
        float *out        = i_t_dm + idm * nsamp;
        fftwf_execute_dft_c2r(plan_c2r, in, out);

        for (unsigned int isamp = 0; isamp < nsamp; isamp++)
        {
            float scale = 1.0f / nsamp;
            float *ptr = i_t_dm + idm * nsamp;
            ptr[isamp] *= scale;
        }
    }

    // Copy output
    std::cout << "Transpose output" << std::endl;
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        for (unsigned int isamp = 0; isamp < nsamp_computed; isamp++)
        {
            float *src_ptr = i_t_dm + idm * nsamp;
            float *dst_ptr = (float *) out + idm * nsamp_computed;
            dst_ptr[isamp] = std::abs(src_ptr[isamp]);
        }
    }

    // Free memory
    free(i_t_nu);
    free(z_f_nu);
    free(z_f_dm);
    free(i_t_dm);
    fftwf_destroy_plan(plan_r2c);
    fftwf_destroy_plan(plan_c2r);
}

} // end namespace dedisp