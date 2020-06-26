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
    // Settings
    float numax = m_f0;
    float dnu   = m_df;

    // Parameters
    unsigned int nchan = m_nchans;   // number of observering frequencies
    unsigned int nsamp = nsamps;     // number of time samples
    unsigned int nfreq = nsamps;     // number of spin frequencies
    unsigned int ndm   = m_dm_count; // number of DMs

    // Transpose input and convert to floating point:
    //  - byte_type input[nsamp][nchan]
    //  -    float f_t_nu[nchan][nsamp]
    std::cout << "Transpose/convert input" << std::endl;
    std::vector<float> f_t_nu(nsamp * nchan);
    for (unsigned int samp = 0; samp < nsamp; samp++)
    {
        for (unsigned int chan = 0; chan < nchan; chan++)
        {
            f_t_nu[chan * nsamp + samp] = (float) in[samp * nchan + chan];
        }
    }

    // Allocate input array (spin vs observing frequency)
    //  - complex float z_f_nu[nfreq][nchan]
    fftwf_complex *z_f_nu = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * nfreq * nchan);

    // Allocate output array (spin frequency vs dm)
    //  - complex float z_f_dm[nfreq][ndm]
    fftwf_complex *z_f_dm = (fftwf_complex *) fftwf_malloc(sizeof(fftwf_complex) * nfreq * ndm);
    memset(z_f_dm, 0, sizeof(fftwf_complex) * nfreq * ndm);

    // FFT data (real to complex) along time axis
    std::cout << "FFT input r2c" << std::endl;
    fftwf_plan plan;
    int rank = 1;
    int n[] = { (int) nsamp };
    int howmany = nchan;
    int idist = n[0], odist = n[0];
    int istride = 1, ostride = 1;
    int *inembed = n, *onembed = n;
    int flags = 0;
    plan = fftwf_plan_many_dft_r2c(
        rank, n, howmany,
        f_t_nu.data(), inembed, istride, idist,
        z_f_nu, onembed, ostride, odist,
        flags);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // Loop over observing frequencies
    for (unsigned int ichan = 0; ichan < nchan; ichan++)
    {
        std::cout << "Processing frequency " << ichan << " / " << nchan << std::endl;
        unsigned int l = nfreq * ichan;
        float nu = numax + (nchan - ichan) * dnu;

        // Loop over DM
        for (unsigned int idm = 0; idm < ndm; idm++)
        {
            float dm = h_dm_list[idm];

            // Loop over spin frequencies
            for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
            {
              float freq = ifreq * dnu;
              int k = ifreq + nfreq * idm;

              // Compute DM delay
              float tdm = dm * (pow(nu, -2) - pow(numax, -2)) / 2.41e-4;

              // Compute phase
              float phase = 2.0 * M_PI * freq * tdm;

              // Compute phasor
              std::complex<float> phasor(cosf(phase), sinf(phase));

              // Complex multiply and add
              z_f_dm[k][0] += z_f_nu[l][0] * phasor.real() - z_f_nu[l][1] * phasor.imag();
              z_f_dm[k][1] += z_f_nu[l][0] * phasor.imag() + z_f_nu[l][1] * phasor.real();
            }
        }
    }

    // Output buffer
    std::vector<float> f_t_dm(nfreq * ndm);
    //  -     float f_t_dm[nfreq][ndm]
    //  - byte_type output[nfreq][ndm]

    // Fourier transform results back to time domain
    std::cout << "FFT input r2c" << std::endl;
    rank = 1;
    n[0] = (int) nfreq;
    howmany = ndm;
    idist = n[0], odist = n[0];
    istride = 1, ostride = 1;
    inembed = n, onembed = n;
    flags = 0;
    plan = fftwf_plan_many_dft_c2r(
        rank, n, howmany,
        z_f_dm, inembed, istride, idist,
        f_t_dm.data(), onembed, ostride, odist,
        flags);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);

    // Convert output to byte_type
    std::cout << "Convert output" << std::endl;
    for (unsigned int freq = 0; freq < nfreq; freq++)
    {
        for (unsigned int dm = 0; dm < ndm; dm++)
        {
            out[freq * ndm + freq] = (byte_type) abs(f_t_dm[freq * ndm + dm]);
        }
    }

    // Free memory
    free(z_f_nu);
    free(z_f_dm);
}

} // end namespace dedisp