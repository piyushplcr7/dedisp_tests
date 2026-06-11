// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
// CPU reference implementations for FDD
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
#ifdef DEDISP_BENCHMARK
    #include "external/Stopwatch.h"
#endif

#include "common/helper.h"
#include "helper.h"
#include "chunk.h"
#include "cufft_optimal_size.hpp" // closestOptimal(), to match FDDGPUPlan FFT sizing

#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include "fitscontainer.hpp" // dataLoader + dataFile (FITS/MPI input for testdedisp_omp)

namespace dedisp
{

// Constructor
FDDCPUPlan::FDDCPUPlan(
    size_type  nchans,
    float_type dt,
    float_type f0,
    float_type df,
    int device_index) :
    Plan(nchans, dt, f0, df, 0)
{
}

// dataLoader-aware constructor (mirrors FDDGPUPlan), for the MPI + FITS driver.
FDDCPUPlan::FDDCPUPlan(const dataLoader& container, int device_index) :
    Plan(container.nchansLocal(),
         container.sampletime(),
         container.f0(),
         container.ddf(),
         container.avgvoverc()),
    nsamps_(container.nsampsLocal()),
    dt_(container.sampletime())
{
    container_ = &container;
}

// Destructor
FDDCPUPlan::~FDDCPUPlan()
{}

// Host-only port of GPUPlan::generate_dm_list_equispaced (no device DM copy).
void FDDCPUPlan::generate_dm_list_equispaced(
    float_type dm_start,
    float_type dm_step,
    size_type  numdms)
{
    if (dm_step <= 0) {
        std::cerr << "Choose a positive dm_step!" << std::endl;
        exit(1);
    }

    h_dm_list.clear();
    for (unsigned int i = 0; i < numdms; ++i) {
        h_dm_list.push_back(dm_start + i * dm_step);
    }

    m_dm_count = h_dm_list.size();
    // Maximum delay in samples (matches GPUPlan)
    m_max_delay = (dedisp_size) roundf(h_dm_list[m_dm_count - 1] * h_delay_table[0]);
}

// Ported from FDDGPUPlan: allocate output_buffer_ and record output sizing.
// Must be called after generating the DM list.
void FDDCPUPlan::setOutputParams(
        bool cleanout,
        bool fftout,
        bool multout,
        int out_nbits,
        char* outfile,
        int w,
        bool barycenter)
{
    cleanout_   = cleanout;
    fftout_     = fftout;
    multout_    = multout;
    outfile_    = outfile;
    w_          = w;
    barycenter_ = barycenter;

    const dedisp_float *dmlist = get_dm_list();
    dedisp_size dm_count = get_dm_count();
    dedisp_size max_delay = get_max_delay();

#ifdef TESTDEDISP_DEBUG
    printf("----------------------------- DM COMPUTATIONS  "
           "----------------------------\n");
    printf("Computing %lu DMs from %f to %f pc/cm^3\n", dm_count, dmlist[0],
           dmlist[dm_count - 1]);
    printf("Max DM delay is %lu samples (%.3f seconds)\n", max_delay,
           max_delay * dt_);
    std::cout << "dt = " << dt_ << std::endl;
#endif

    // FFT size mirrors FDDGPUPlan::setOutputParams (closestOptimal padding).
    nsamp_fft_    = closestOptimal(nsamps_ + max_delay, true);
    nsamp_padded_ = 2ULL * (nsamp_fft_ / 2 + 1);

    if (cleanout_) {
        nsamps_computed_ = nsamps_ - max_delay;
    } else {
        nsamps_computed_ = nsamps_ + max_delay;
    }

    if (fftout) {
        output_buffer_ = std::make_unique<float[]>(nsamp_padded_ * dm_count);
    } else {
        output_buffer_ = std::make_unique<float[]>(nsamps_computed_ * dm_count);
    }

    if (output_buffer_ == nullptr) {
        printf("\nERROR: Failed to allocate output array\n");
        exit(-1);
    }
}

// Ported from FDDGPUPlan: write per-DM timeseries/FFT outputs (host file I/O).
void FDDCPUPlan::writeOutput(char* outfile, int w, bool barycenter, const std::vector<int>& inForOut)
{
    std::cout << "\n----------------------------- WRITING OUTPUT ----------------------------\n" << std::endl;
    const char* outfiles_basename = (outfile == NULL) ? "output" : outfile;
    auto start_time = std::chrono::high_resolution_clock::now();

    unsigned dm_count = m_dm_count;
    const float* dmlist = get_dm_list();
    unsigned out_nbits = 32;
    float* output = output_buffer_.get();

    if (multout_ && !fftout_ && barycenter) {
        int Nout = inForOut.size();
        std::cout << "Writing barycentered timeseries" << std::endl;

        for (int i = 0 ; i < (int)inForOut.size() ; ++i) {
            if (inForOut[i] >= (int)nsamps_computed_) {
                Nout = i;
                break;
            }
        }

        #pragma omp parallel
        {
            float* barycentered_data = new float[Nout];
            #pragma omp for
            for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
                char out_file_name[256];
                sprintf(out_file_name,"%s_DM%.*f.%s", outfiles_basename, w, dmlist[out_file_idx], "dat");

                float* dedispersed_data = output + out_file_idx * (size_t)nsamps_computed_;
                for (int i = 0 ; i < Nout ; ++i) {
                    barycentered_data[i] = dedispersed_data[inForOut[i]];
                }

                int fd = open(out_file_name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) { std::cerr << "Open failed\n"; std::exit(1); }
                size_t numtowrite = (size_t)Nout * out_nbits / 8;
                ssize_t written = write(fd, barycentered_data, numtowrite);
                if (written != (ssize_t)numtowrite) { std::cerr << "Write failed\n"; std::exit(1); }
                close(fd);
            }
            delete[] barycentered_data;
        }
    }
    else if (multout_ && !fftout_ && !barycenter) {
        #pragma omp parallel
        {
            #pragma omp for
            for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
                char out_file_name[256];
                sprintf(out_file_name,"%s_DM%.*f.%s", outfiles_basename, w, dmlist[out_file_idx], "dat");

                int fd = open(out_file_name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd < 0) { std::cerr << "Open failed\n"; std::exit(1); }
                // Each output row holds the full (dirty) dedispersed buffer of
                // stride nsamps_computed_ = nsamps + max_delay. Skip the first
                // dm_offset samples and write nsamps - max_delay clean samples.
                // dm_offset is the PER-DM dispersion delay (roundf(dm*delay[0])),
                // not the global max_delay: this places each DM's timeseries at
                // its own true time origin so cpuout aligns bit-for-bit with the
                // reference (masterout). Mirrors the reference convention noted in
                // FDDGPUPlan.cpp:148. Runs WITHOUT -cleanout, like the GPU.
                size_t numtowrite = (nsamps_ - get_max_delay()) / 2 * 2;
                size_t bytestowrite = numtowrite * out_nbits / 8;
                size_t dm_offset = (size_t) roundf(dmlist[out_file_idx] * h_delay_table[0]);
                ssize_t written = write(fd,
                    output + out_file_idx * (size_t)nsamps_computed_ + dm_offset,
                    bytestowrite);
                if (written != (ssize_t)bytestowrite) { std::cerr << "Write failed\n"; std::exit(1); }
                close(fd);
            }
        }
    }
    else if (multout_ && fftout_) {
        #pragma omp parallel for
        for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
            char out_file_name[256];
            sprintf(out_file_name,"%s_DM%.*f.%s", outfiles_basename, w, dmlist[out_file_idx], "fft");

            FILE* file_out = fopen(out_file_name, "wb");
            size_t numtowrite = (size_t)nsamp_padded_ * out_nbits / 8;
            size_t writtennum = fwrite(output + out_file_idx * (size_t)nsamp_padded_, 1, numtowrite, file_out);
            if (writtennum != numtowrite) {
                std::cerr << "Writing file " << out_file_idx << " failed!" << std::endl;
            }
            fclose(file_out);
        }
    }
    else {
        FILE *file_out;
        char out_file_name[256];
        if (fftout_) {
            sprintf(out_file_name,"%s.allDMs.fft", outfiles_basename);
            file_out = fopen(out_file_name, "wb");
        } else {
            sprintf(out_file_name,"%s.allDMs.dat", outfiles_basename);
            file_out = fopen(out_file_name, "wb");
        }
        fwrite(output, 1, (size_t)(fftout_? nsamp_padded_ : nsamps_computed_) * dm_count * out_nbits / 8, file_out);
        fclose(file_out);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    std::cout << "Writing the output took " << (double)duration_us / 1e6 << " seconds" << std::endl;
}

// Ported from FDDGPUPlan: write a PRESTO-style .inf metadata file per DM.
void FDDCPUPlan::writeInfs(char* outfile, const dataFile* file, size_t nsamps, double dt, int w,
                           bool barycenter, double blotoa, double avgvoverc)
{
    const char* outfiles_basename = (outfile == NULL) ? "output" : outfile;

    unsigned dm_count = m_dm_count;
    const float* dmlist = get_dm_list();

    if (!multout_)
        return;

    #pragma omp parallel for
    for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
        char out_inf_name[256];
        sprintf(out_inf_name,"%s_DM%.*f.inf", outfiles_basename, w, dmlist[out_file_idx]);

        FILE* inf_out = fopen(out_inf_name,"w");

        fprintf(inf_out,"%-40s=  %s_DM_%.*f\n", " Data file name without suffix", outfiles_basename, w, dmlist[out_file_idx]);
        fprintf(inf_out,"%-40s=  %s\n", " Telescope used", file->telescope());
        fprintf(inf_out,"%-40s=  %s\n", " Instrument used", file->instrument());
        fprintf(inf_out,"%-40s=  %s\n", " Object being observed", file->objectName());
        fprintf(inf_out,"%-40s=  %s\n", " J2000 Right Ascension (hh:mm:ss.ssss)", file->rightAscension());
        fprintf(inf_out,"%-40s=  %s\n", " J2000 Declination     (dd:mm:ss.ssss)", file->declination()[0] == '+' ? file->declination() + 1 : file->declination());
        fprintf(inf_out,"%-40s=  %s\n", " Data observed by", file->observer());
        double epoch;
        if (barycenter) {
            double baryhifreq = file->freqs().back() * (1.0 + avgvoverc);
            double barydispdt = dmlist[out_file_idx] / (0.000241 * baryhifreq * baryhifreq);
            epoch = blotoa - barydispdt / 86400.0;
        } else {
            epoch = file->epoch();
        }
        fprintf(inf_out,"%-40s=  %.15f\n", " Epoch of observation (MJD)", epoch);
        fprintf(inf_out,"%-40s=  %d\n", " Barycentered?           (1 yes, 0 no)", barycenter ? 1 : 0);
        fprintf(inf_out,"%-40s=  %ld\n", " Number of bins in the time series", nsamps);
        fprintf(inf_out,"%-40s=  %.4f\n", " Width of each time series bin (sec)", dt);
        fprintf(inf_out,"%-40s=  1\n", " Any breaks in the data? (1 yes, 0 no)");
        fprintf(inf_out,"%-40s=  0, %ld\n", " On/Off bin pair #  1 ", nsamps-1);
        fprintf(inf_out,"%-40s=  %ld, %ld\n", " On/Off bin pair #  2", nsamps-1, nsamps-1);
        fprintf(inf_out,"%-40s=  Radio\n", " Type of observation (EM band)  ");
        fprintf(inf_out,"%-40s=  900\n", " Beam diameter (arcsec)");
        fprintf(inf_out,"%-40s=  %.*f\n", " Dispersion measure (cm-3 pc)", w, dmlist[out_file_idx]);
        fprintf(inf_out,"%-40s=  %.7f\n", " Central freq of low channel (MHz)", file->freqs()[0]);
        fprintf(inf_out, "%-40s=  %.7f\n", " Total bandwidth (MHz)", file->bw());
        fprintf(inf_out, "%-40s=  %d\n", " Number of channels", file->nchan());
        fprintf(inf_out, "%-40s=  %.15f\n", " Channel bandwidth (MHz)", -file->ddf());

        char *user = getenv("USER");
        if (!user) user = getenv("USERNAME");
        fprintf(inf_out, "%-40s=  %s\n", " Data analyzed by", user ? user : "Unknown");
        fprintf(inf_out, " Any additional notes: \n \tProject ID %s, Date: %s.\n \t4 polns were not summed.  Samples have 8 bits. \n", file->projid(), file->dateobs());

        fclose(inf_out);
    }
}

// Dedisperison kernels

// Reference implementation for FDD on CPU
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
            tdms[ichan] = roundf(dms[idm] * delays[ichan]); // integer-sample delay, matches GPU kernel
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

// Reference implementation for FDD on CPU with time segmentation feature
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
            tdms[ichan] = roundf(dms[idm] * delays[ichan]); // integer-sample delay, matches GPU kernel
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

// Reference implementation for FDD on CPU with time segmentation feature
// Optimized to process batches of channels in parallel
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
    // Transpose input data. Heap-allocated (not stack VLAs) and channel-padded
    // to a multiple of nchan_batch (=32), flattened to [irow * nchan_padded +
    // ichan]. Padded channels are zero so the batched loop strides cleanly.
    const unsigned int nchan_batch_pad = 32;
    const unsigned int nchan_padded    = round_up(nchan, nchan_batch_pad);
    const size_t       nrows           = (size_t) nchunk * nfreq_chunk_padded;
    std::vector<float> in_real(nrows * nchan_padded, 0.0f);
    std::vector<float> in_imag(nrows * nchan_padded, 0.0f);
    #pragma omp parallel for
    for (unsigned int ifreq = 0; ifreq < nchunk * nfreq_chunk_padded; ifreq++)
    {
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            auto* sample = &in[ichan * in_stride];
            in_real[(size_t) ifreq * nchan_padded + ichan] = sample[ifreq].real();
            in_imag[(size_t) ifreq * nchan_padded + ichan] = sample[ifreq].imag();
        }
    }

    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        // DM delays (padded to nchan_padded; tail delays 0 -> no contribution)
        float tdms[nchan_padded];
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            tdms[ichan] = roundf(dms[idm] * delays[ichan]); // integer-sample delay, matches GPU kernel
        }
        for (unsigned int ichan = nchan; ichan < nchan_padded; ichan++)
        {
            tdms[ichan] = 0.0f;
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

                for (unsigned int ichan_outer = 0; ichan_outer < nchan_padded; ichan_outer += nchan_batch)
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
                        float sample_real = in_real[(size_t)(ichunk * nfreq_chunk_padded + ifreq) * nchan_padded + ichan];
                        float sample_imag = in_imag[(size_t)(ichunk * nfreq_chunk_padded + ifreq) * nchan_padded + ichan];

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

/* Reference implementation for FDD on CPU, used by default for CPU execution
* Optimized to process batches of channels in parallel
* Optinally enable extrapolation of phasor computations
* by setting the last templated argument to true.
* With this feature enabled, extrapolation is used in the computation of the phasors
* On CPU this feature provides a noticable improvement,
* but on GPU this feature only provides a marginal improvement.
* Boudary conditions should be further explored to determine
* functional correctness at all times.
* Leaving this feature in because it might be beneficial to use.
*/
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
    // Transpose input data. Heap-allocated (not stack VLAs): with real data
    // nfreq is ~5e5, so float[nfreq][nchan] would overflow the stack. Flattened
    // to [ifreq * nchan_padded + ichan]. The channel dimension is padded up to a
    // multiple of nchan_batch and zero-filled, so the batched dedispersion loop
    // below can stride to nchan_padded without a tail guard: padded channels
    // have sample 0 (and tdms 0 -> phasor (1,0)), contributing nothing.
    const unsigned int nchan_padded = round_up(nchan, nchan_batch);
    std::vector<float> in_real((size_t) nfreq * nchan_padded, 0.0f);
    std::vector<float> in_imag((size_t) nfreq * nchan_padded, 0.0f);
    #pragma omp parallel for
    for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
    {
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            auto* sample = &in[ichan * in_stride];
            in_real[(size_t) ifreq * nchan_padded + ichan] = sample[ifreq].real();
            in_imag[(size_t) ifreq * nchan_padded + ichan] = sample[ifreq].imag();
        }
    }

    #pragma omp parallel for
    for (unsigned int idm = 0; idm < ndm; idm++)
    {
        // DM delays (padded to nchan_padded; tail delays 0 so padded channels
        // produce phasor (1,0) against a zero sample -> no contribution).
        float tdms[nchan_padded];
        for (unsigned int ichan = 0; ichan < nchan; ichan++)
        {
            tdms[ichan] = roundf(dms[idm] * delays[ichan]); // integer-sample delay, matches GPU kernel
        }
        for (unsigned int ichan = nchan; ichan < nchan_padded; ichan++)
        {
            tdms[ichan] = 0.0f;
        }

        // Loop over spin frequencies
        for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
        {
            // Partial sums
            float sums_real[nchan_batch];
            float sums_imag[nchan_batch];
            memset(sums_real, 0, nchan_batch * sizeof(float));
            memset(sums_imag, 0, nchan_batch * sizeof(float));

            for (unsigned int ichan_outer = 0; ichan_outer < nchan_padded; ichan_outer += nchan_batch)
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
                }
                else // No extrapolation
                {
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
                }

                // Loop over observing frequencies
                for (unsigned int ichan_inner = 0; ichan_inner < nchan_batch; ichan_inner++)
                {
                    unsigned int ichan = ichan_outer + ichan_inner;

                    // Load sample
                    float sample_real = in_real[(size_t) ifreq * nchan_padded + ichan];
                    float sample_imag = in_imag[(size_t) ifreq * nchan_padded + ichan];

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

/* Public interface for FDD CPU implementation
*  By default runs the DM batched optimized implementation of FDD on CPU.
*  Set environment variable USE_SEGMENTED to use alternative time segmentation feature.
*  Set environment variable USE_REFERENCE to use the straight forward
*  non-optimized implementation of FDD, might be combined with USE_SEGMENTED.
*  Refer to the GPU implementation in GPU source files for more elaborate annotations.
*  The CPU implementation uses FFTW for the forwards and backwards FFTs.
*/
void FDDCPUPlan::execute(
    size_type        nsamps,
    const byte_type* in,
    size_type        in_nbits,
    byte_type*       out,
    size_type        out_nbits,
    unsigned         flags)
{
    char* use_segmented_str = getenv("USE_SEGMENTED");
    bool use_segmented = !use_segmented_str ? false : atoi(use_segmented_str);
    if (use_segmented)
    {
#ifdef DEDISP_DEBUG
        std::cout << ">> Running segmented CPU implementation" << std::endl;
#endif
        execute_cpu_segmented(nsamps, in, in_nbits, out, out_nbits);
    } else { //Default
#ifdef DEDISP_DEBUG
        std::cout << ">> Running CPU implementation" << std::endl;
#endif
        execute_cpu(nsamps, in, in_nbits, out, out_nbits);
    }
}


// Private interface to CPU implementation of FDD
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
    unsigned int ndm   = m_dm_count;    // number of DMs

    // Compute the number of output samples
    unsigned int nsamp_computed = nsamp - m_max_delay;

    // FFT length: zero-pad to (nsamp + max_delay) to avoid dedispersion
    // wrap-around, rounded up to an FFT-optimal size (factors of 2,3,5,7).
    // This mirrors FDDGPUPlan::execute_gpu (closestOptimal + nsamp_padded).
    unsigned int nsamp_fft    = closestOptimal(nsamp + m_max_delay, true);
    unsigned int nsamp_padded = 2 * (nsamp_fft / 2 + 1);
    unsigned int nfreq        = (nsamp_fft / 2 + 1); // number of spin frequencies
#ifdef DEDISP_DEBUG
    std::cout << debug_str << std::endl;
    std::cout << "nsamp_fft    = " << nsamp_fft << std::endl;
    std::cout << "nsamp_padded = " << nsamp_padded << std::endl;
#endif

    // Timers
#ifdef DEDISP_BENCHMARK
    std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();
#endif

    // Allocate memory
#ifdef DEDISP_DEBUG
    std::cout << memory_alloc_str << std::endl;
#endif
    std::vector<float> data_nu;
    std::vector<float> data_dm;
    data_nu.resize((size_t) nchan * nsamp_padded);
    data_dm.resize((size_t) ndm * nsamp_padded);

    // Generate spin frequency table
    if (h_spin_frequencies.size() != nfreq)
    {
        generate_spin_frequency_table(nfreq, nsamp_fft, dt);
    }
#ifdef DEDISP_BENCHMARK
    init_timer->Pause();
#endif

    // Transpose input and convert to floating point
#ifdef DEDISP_DEBUG
    std::cout << prepare_input_str << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
    preprocessing_timer->Start();
#endif
    // Honor in_nbits like execute_gpu does, matching the two GPU kernels exactly:
    //   in_nbits == 8  -> transpose_unpack_kernel: result = (val - 127.5f) * (1/nchan)
    //   in_nbits == 32 -> transpose_kernel<float>: PURE transpose, out = in
    //                     (the `scale` arg is passed but ignored; NO offset).
    // The dataLoader's assembled buffer is float[] of already-reduced samples
    // ((raw*scl+offs)*wt from reduceData), so the MPI/FITS driver runs with
    // in_nbits == 32. The GPU treats that buffer as-is, so the CPU MUST NOT
    // re-subtract 127.5 or re-divide by nchan -- doing so squashes every spike
    // by ~nchan and pins the output to a -127.5 DC floor (the "baseline matches
    // but spikes vanish" symptom). Use offset 0, scale 1 here.
    // out: input at offset 0, tail [nsamp, nsamp_padded) stays zero (data_nu is
    // zero-initialized). Matches FDDGPUPlan: transpose at 0 + tail zero-pad (NOT
    // a front-shift). The clean region is recovered by the +m_max_delay output
    // offset below.
    if (in_nbits == 8) {
        transpose_data<const byte_type, float>(
            nchan, nsamp, nchan, nsamp_padded, 127.5, nchan,
            in, data_nu.data());
    } else { // in_nbits == 32: assembled float buffer, pure transpose (no offset/scale)
        transpose_data<const float, float>(
            nchan, nsamp, nchan, nsamp_padded, 0.0f, 1.0f,
            reinterpret_cast<const float*>(in), data_nu.data());
    }

    // FFT data (real to complex) along time axis
#ifdef DEDISP_DEBUG
    std::cout << fft_r2c_str << std::endl;
#endif
    fft_r2c_inplace(
        nsamp_fft,       // n
        nchan,           // batch
        nsamp_padded,    // stride
        data_nu.data()); // data
#ifdef DEDISP_BENCHMARK
    preprocessing_timer->Pause();
#endif

    // Dedispersion in frequency domain
#ifdef DEDISP_DEBUG
    std::cout << fdd_dedispersion_str << std::endl;
#endif

    char* use_reference_str = getenv("USE_REFERENCE");
    bool use_reference = !use_reference_str ? false : atoi(use_reference_str);
    if (use_reference)
    {
#ifdef DEDISP_DEBUG
        std::cout << ">> Running reference implementation" << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
        dedispersion_timer->Start();
#endif
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
#ifdef DEDISP_DEBUG
        std::cout << ">> Running optimized implementation" << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
        dedispersion_timer->Start();
#endif
        // Set last templated parameter to true to enable extrapolation feature
        dedisperse_optimized<float, float, 32, false>(
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
#ifdef DEDISP_BENCHMARK
    dedispersion_timer->Pause();
#endif

    // Fourier transform results back to time domain
#ifdef DEDISP_DEBUG
    std::cout << fft_c2r_str << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
    postprocessing_timer->Start();
#endif
    fft_c2r_inplace(
        nsamp_fft,    // n
        ndm,          // batch
        nsamp_padded, // stride
        data_dm.data());
#ifdef DEDISP_BENCHMARK
    postprocessing_timer->Pause();
#endif

    // Copy output
#ifdef DEDISP_DEBUG
    std::cout << copy_output_str << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
    output_timer->Start();
#endif
    // Mirror FDDGPUPlan::execute_gpu exactly: copy nsamps_computed_ samples per
    // DM from data_dm[0] -- the FULL "dirty" buffer, including the first max_delay
    // wrap-corrupted samples. writeOutput() then skips those max_delay samples.
    // This is why the program runs WITHOUT -cleanout: setOutputParams sets
    // nsamps_computed_ = nsamps + max_delay in that mode, so each output row is
    // long enough to hold the dirty buffer and let writeOutput offset into it.
    // (The synthetic driver, which never calls setOutputParams, leaves
    // nsamps_computed_ == 0 and falls back to the clean-region extraction.)
    size_t out_rowlen;
    float* copy_src;
    if (nsamps_computed_ > 0) {
        out_rowlen = nsamps_computed_; // = nsamp + m_max_delay when !cleanout
        copy_src   = data_dm.data();   // dirty buffer from offset 0, matches GPU
    } else {
        out_rowlen = nsamp_computed;             // clean region (synthetic path)
        copy_src   = data_dm.data() + m_max_delay;
    }
    copy_data<float, float>(
        ndm, out_rowlen,          // height, width
        nsamp_padded, out_rowlen, // in stride, out stride
        copy_src,                 // input
        (float *) out);
#ifdef DEDISP_BENCHMARK
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
#endif
}

// Private interface to CPU implementation of FDD using time segmentation feature
void FDDCPUPlan::execute_cpu_segmented(
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
    unsigned int nfft  = 16384;         // number of samples processed in a segment

    // Compute the number of output samples
    unsigned int nsamp_computed = nsamp - m_max_delay;

    // Compute the number of chunks
    unsigned int nsamp_dm   = std::ceil(m_max_delay);
    float min_efficiency    = 0.8;
    while ((nfft * (1.0 - min_efficiency)) < nsamp_dm) { nfft *= 2; };
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
#ifdef DEDISP_DEBUG
    std::cout << debug_str << std::endl;
    std::cout << "nfft               = " << nfft << std::endl;
    std::cout << "nsamp_dm           = " << nsamp_dm << std::endl;
    std::cout << "nsamp_good         = " << nsamp_good << std::endl;
    std::cout << "nchunk             = " << nchunk << std::endl;
    std::cout << "nfreq_chunk        = " << nfreq_chunk << std::endl;
    std::cout << "nfreq_chunk_padded = " << nfreq_chunk_padded << std::endl;
    std::cout << "nsamp_padded       = " << nsamp_padded << std::endl;
#endif

    // Timers
#ifdef DEDISP_BENCHMARK
    std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
    std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
    total_timer->Start();
    init_timer->Start();
#endif

    // Allocate memory
#ifdef DEDISP_DEBUG
    std::cout << memory_alloc_str << std::endl;
#endif
    std::vector<float> data_t_nu((size_t) nchan * nsamp_padded);
    std::vector<std::complex<float>> data_f_nu((size_t) nchan * nsamp_padded/2);

    // Compute chunks
    std::vector<Chunk> chunks(nchunk);
    unsigned int nfreq_computed;
    compute_chunks(
        nsamp, nsamp_good, nfft,
        nfreq_chunk_padded, nfreq_computed, chunks);

    // Generate spin frequency table
    if (h_spin_frequencies.size() != nsamp_padded)
    {
        h_spin_frequencies.resize(nsamp_padded);
        generate_spin_frequency_table_chunks(
            chunks, h_spin_frequencies,
            nfreq_chunk, nfreq_chunk_padded,
            nfft, dt);
    }
#ifdef DEDISP_BENCHMARK
    init_timer->Pause();
#endif

    // Transpose input and convert to floating point:
#ifdef DEDISP_DEBUG
    std::cout << prepare_input_str << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
    preprocessing_timer->Start();
#endif
    transpose_data<const byte_type, float>(
        nchan,             // height
        nsamp,             // width
        nchan,             // in_stride
        nsamp_padded,      // out_stride
        127.5,             // offset
        nchan,             // scale
        in,                // in
        data_t_nu.data()); // out

    // Debug
#ifdef DEDISP_DEBUG
    std::cout << debug_str << std::endl;
    print_chunks(chunks);
    std::cout << "nfreq_computed = " << nfreq_computed << std::endl;
    std::cout << "nsamp_computed = " << nsamp_computed << std::endl;
#endif

    // FFT data (real to complex) along time axis
#ifdef DEDISP_DEBUG
    std::cout << fft_r2c_str << std::endl;
#endif
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
#ifdef DEDISP_BENCHMARK
    preprocessing_timer->Pause();
#endif

    // Free input buffer
    data_t_nu.resize(0);

    // Allocate buffers
#ifdef DEDISP_DEBUG
    std::cout << memory_alloc_str << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
    init_timer->Start();
#endif
    std::vector<std::complex<float>> data_f_dm((size_t) ndm * nsamp_padded/2);
    std::vector<float> data_t_dm((size_t) ndm * nsamp_padded);
#ifdef DEDISP_BENCHMARK
    init_timer->Pause();
#endif

    // Perform dedispersion
#ifdef DEDISP_DEBUG
    std::cout << fdd_dedispersion_str << std::endl;
#endif
    char* use_reference_str = getenv("USE_REFERENCE");
    bool use_reference = !use_reference_str ? false : atoi(use_reference_str);
    char* use_segmented_kernel_str = getenv("USE_SEGMENTED_KERNEL");
    bool use_segmented_kernel = !use_segmented_kernel_str ? false : atoi(use_segmented_kernel_str);
    if (use_reference)
    {
#ifdef DEDISP_BENCHMARK
        dedispersion_timer->Start();
#endif
        if (use_segmented_kernel)
        {
#ifdef DEDISP_DEBUG
            std::cout << ">> Running segmented reference kernel" << std::endl;
#endif
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
        } else {
#ifdef DEDISP_DEBUG
            std::cout << ">> Running reference kernel" << std::endl;
#endif
            dedisperse_reference<float, float>(
                ndm, nfreq, nchan,         // data dimensions
                dt,                        // sample time
                h_spin_frequencies.data(), // spin frequencies
                h_dm_list.data(),          // DMs
                h_delay_table.data(),      // delay table
                nsamp_padded/2,            // in stride
                nsamp_padded/2,            // out stride
                data_f_nu.data(),          // input
                data_f_dm.data()           // output
            );
        }
    }
    else
    {
#ifdef DEDISP_BENCHMARK
        dedispersion_timer->Start();
#endif
        if (use_segmented_kernel)
        {
#ifdef DEDISP_DEBUG
            std::cout << ">> Running segmented optimized kernel" << std::endl;
#endif
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
        } else {
#ifdef DEDISP_DEBUG
            std::cout << ">> Running optimized kernel" << std::endl;
#endif
            // Set last templated parameter to true to enable extrapolation feature
            dedisperse_optimized<float, float, 32, false>(
                ndm, nfreq, nchan,         // data dimensions
                dt,                        // sample time
                h_spin_frequencies.data(), // spin frequencies
                h_dm_list.data(),          // DMs
                h_delay_table.data(),      // delay table
                nsamp_padded/2,            // in stride
                nsamp_padded/2,            // out stride
                data_f_nu.data(),          // input
                data_f_dm.data()           // output
            );
        }
    }
#ifdef DEDISP_BENCHMARK
    dedispersion_timer->Pause();
#endif

    // Fourier transform results back to time domain
#ifdef DEDISP_DEBUG
    std::cout << fft_c2r_str << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
    postprocessing_timer->Start();
#endif
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
    #pragma omp parallel for
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
#ifdef DEDISP_BENCHMARK
    postprocessing_timer->Pause();
#endif

    // Copy output
#ifdef DEDISP_DEBUG
    std::cout << copy_output_str << std::endl;
#endif
#ifdef DEDISP_BENCHMARK
    output_timer->Start();
#endif
    copy_chunk_output(
        data_t_dm.data(), (float *) out,
        ndm, nsamp, nsamp_computed,
        nsamp_padded, nsamp_good, chunks);
#ifdef DEDISP_BENCHMARK
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
#endif
}

// Private helper functions
void FDDCPUPlan::generate_spin_frequency_table(
    dedisp_size nfreq,
    dedisp_size nsamp_fft,
    dedisp_float dt)
{
    h_spin_frequencies.resize(nfreq);

    #pragma omp parallel for
    for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++)
    {
        // Skipping the 1/dt scaling (absorbed by using rounded sample delays in
        // the dedispersion phase), matching FDDGPUPlan::generate_spin_frequency_table.
        h_spin_frequencies[ifreq] = ifreq * (1.0 / nsamp_fft);
    }
}

// Wrapper for FFTW R2C FFT (in -> out)
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

// Wrapper for FFTW R2C FFT (inplace)
void FDDCPUPlan::fft_r2c_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    fft_r2c(n, batch, stride, stride, data, data);
}

// Wrapper for FFTW C2R FFT (in -> out)
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

// Wrapper for FFTW C2R FFT (inplace)
void FDDCPUPlan::fft_c2r_inplace(
    unsigned int n,
    unsigned int batch,
    size_t stride,
    float *data)
{
    fft_c2r(n, batch, stride, stride, data, data);
}

} // end namespace dedisp