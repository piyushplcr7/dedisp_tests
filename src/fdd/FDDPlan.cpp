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
    unsigned int ndm_batch_max = 16;
    unsigned int ndm_fft_batch = 16;
    unsigned int ndm_buffers   = 4;

    // Maximum number of channels processed in one gulp
    unsigned int nchan_batch_max = 32;
    unsigned int nchan_fft_batch = 32;
    unsigned int nchan_buffers   = 2;

    // Compute derived counts
    dedisp_size out_bytes_per_sample = out_nbits / (sizeof(dedisp_byte) *
                                                    BITS_PER_BYTE);
    dedisp_size chans_per_word = sizeof(dedisp_word)*BITS_PER_BYTE / in_nbits;

    // The number of channel words in the input
    dedisp_size nchan_words = nchan / chans_per_word;

    // The number of channel words proccessed in one gulp
    dedisp_size nchan_words_gulp = nchan_batch_max / chans_per_word;

    // Allocate host memory
    cu::HostMemory   h_data_dm(ndm * nsamp_padded * sizeof(float));

    // Allocate device memory
    cu::DeviceMemory d_data_nu(nchan_batch_max * nsamp_padded * sizeof(float));

    // Buffers for double buffering
    std::vector<cu::HostMemory> h_data_in_(nchan_buffers);
    std::vector<cu::DeviceMemory> d_data_in_(nchan_buffers);
    std::vector<cu::DeviceMemory> d_data_out_(ndm_buffers);
    for (unsigned int i = 0; i < nchan_buffers; i ++)
    {
        h_data_in_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
        d_data_in_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
    }
    for (unsigned int i = 0; i < ndm_buffers; i ++)
    {
        d_data_out_[i].resize(ndm_batch_max * nsamp_padded * sizeof(float));
    }

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
        nchan_fft_batch);       // batch
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
        ndm_fft_batch);         // batch
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

    // Initialize FDDKernel
    FDDKernel kernel;
    kernel.copy_delay_table(
        d_delay_table,
        m_nchans * sizeof(dedisp_float),
        0, *htodstream);

    struct ChannelData
    {
        unsigned int ichan_start;
        unsigned int nchan_batch_current;
        void* h_in_ptr;
        void* d_in_ptr;
        cu::Event inputStart, inputEnd;
        cu::Event computeStart, computeEnd;
        cu::Event outputStart, outputEnd;
    };

    unsigned int nchan_jobs = (nchan + nchan_batch_max) / nchan_batch_max;
    std::vector<ChannelData> channel_jobs(nchan_jobs);

    for (unsigned job_id = 0; job_id < nchan_jobs; job_id++)
    {
        ChannelData& job = channel_jobs[job_id];
        job.ichan_start        = job_id == 0 ? 0 : channel_jobs[job_id - 1].ichan_start
                                 + nchan_batch_max;
        job.nchan_batch_current = std::min(nchan_batch_max, nchan - job.ichan_start);
        job.h_in_ptr           = h_data_in_[job_id % nchan_buffers];
        job.d_in_ptr           = d_data_in_[job_id % nchan_buffers];
        if (job.nchan_batch_current == 0) {
            channel_jobs.pop_back();
        }
    }

    struct DMData{
        unsigned int idm_start;
        unsigned int ndm_batch_current;
        float* h_out_ptr;
        dedisp_float2* d_out_ptr;
        cu::Event inputStart, inputEnd;
        cu::Event computeStart, computeEnd;
        cu::Event outputStart, outputEnd;
    };

    unsigned int ndm_jobs = (ndm + ndm_batch_max) / ndm_batch_max;
    std::vector<DMData> dm_jobs(ndm_jobs);

    for (unsigned job_id = 0; job_id < ndm_jobs; job_id++)
    {
        DMData& job = dm_jobs[job_id];
        job.idm_start = job_id == 0 ? 0 : dm_jobs[job_id - 1].idm_start
                        + ndm_batch_max;
        job.ndm_batch_current = std::min(ndm_batch_max, ndm - job.idm_start);
        job.d_out_ptr         = d_data_out_[job_id % ndm_buffers];
        if (job.ndm_batch_current == 0)
        {
            dm_jobs.pop_back();
        }
    }

    std::cout << "Perform dedispersion in frequency domain" << std::endl;

    // Process all dm batches
    for (unsigned dm_job_id_outer = 0; dm_job_id_outer < dm_jobs.size(); dm_job_id_outer += ndm_buffers)
    {
        // Initialize output to zero
        for (cu::DeviceMemory& d_data_out : d_data_out_)
        {
            d_data_out.zero(*htodstream);
        }

        // Info
        auto& dm_job_first = dm_jobs[dm_job_id_outer];
        std::cout << "Processing DM " << dm_job_first.idm_start;
        if (dm_job_id_outer + 1 < ndm){
            auto& dm_job_second = dm_jobs[dm_job_id_outer+1];
            auto idm_end = dm_job_second.idm_start + dm_job_second.ndm_batch_current;
            std::cout << " to " << idm_end + 1 << std::endl;
        } else {
            std::cout << "." << std::endl;
        }

        // Process all channel batches
        for (unsigned channel_job_id = 0; channel_job_id < channel_jobs.size(); channel_job_id++)
        {
            auto& channel_job = channel_jobs[channel_job_id];

            // Compute current number of channels
            unsigned int ichan_end = channel_job.ichan_start + channel_job.nchan_batch_current;
            unsigned int nchan_current = std::min(nchan_batch_max, nchan - channel_job.ichan_start);

            // Info
            std::cout << "Processing channel " << channel_job.ichan_start << " to " << ichan_end << std::endl;

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
            htodstream->synchronize();

            // Transpose and upack the data
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

            // FFT data (real to complex) along time axis
            for (unsigned int i = 0; i < nchan_batch_max/nchan_fft_batch; i++)
            {
                cufftReal    *idata = (cufftReal *) d_data_nu.data() + i * nsamp_padded * nchan_fft_batch;
                cufftComplex *odata = (cufftComplex *) idata;
                cufftExecR2C(plan_r2c, idata, odata);
            }

            // Process DM batches
            for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
            {
                unsigned int dm_job_id = dm_job_id_outer + dm_job_id_inner;
                if (dm_job_id >= dm_jobs.size())
                {
                    break;
                }
                auto& dm_job = dm_jobs[dm_job_id];

                // Compute current number of channels
                unsigned int ndm_current = std::min(ndm_batch_max, ndm - dm_job.idm_start);
                unsigned int idm_end = dm_job.idm_start + dm_job.ndm_batch_current;

                // Info
                std::cout << "Processing DM " << dm_job.idm_start << " to " << idm_end << std::endl;

                // Wait for temporary output from previous job to be copied
                if (channel_job_id > (nchan_buffers-1))
                {
                    auto& job_previous = channel_jobs[channel_job_id - nchan_buffers];
                    job_previous.outputEnd.synchronize();
                }

                // Dedispersion in frequency domain
                kernel.launch(
                    ndm_current,             // ndm
                    nfreq,                   // nfreq
                    nchan_current,           // nchan
                    dt,                      // dt
                    d_spin_frequencies,      // d_spin_frequencies
                    d_dm_list,               // d_dm_list
                    d_data_nu,               // d_in
                    dm_job.d_out_ptr,        // d_out
                    nsamp_padded/2,          // in stride
                    nsamp_padded/2,          // out stride
                    dm_job.idm_start,        // idm_start
                    channel_job.ichan_start, // ichan_start
                    *executestream);         // stream
                executestream->record(dm_job.computeEnd);
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
        } // end for ichan_start

        // Output DM batches
        for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++)
        {
            unsigned int dm_job_id = dm_job_id_outer + dm_job_id_inner;
            if (dm_job_id >= dm_jobs.size())
            {
                break;
            }
            auto& dm_job = dm_jobs[dm_job_id];
            unsigned int ndm_gulp_current = std::min(ndm_batch_max, ndm - dm_job.idm_start);

            // Get pointer to DM output data on host and on device
            dedisp_size dm_stride = nsamp_padded * out_bytes_per_sample;
            dedisp_size dm_offset = dm_job.idm_start * dm_stride;
            auto* h_out = (void *) (((size_t) h_data_dm.data()) + dm_offset);
            auto *d_out = (float *) dm_job.d_out_ptr;

            // Fourier transform results back to time domain
            for (unsigned int i = 0; i < ndm_batch_max/ndm_fft_batch; i++)
            {
                cufftReal    *odata = (cufftReal *) d_out + i * nsamp_padded * ndm_fft_batch;
                cufftComplex *idata = (cufftComplex *) odata;
                cufftExecC2R(plan_c2r, idata, odata);
            }

            // FFT scaling
            kernel.scale(
                ndm_gulp_current, // height
                nsamp_computed,   // width
                nsamp_padded,     // stride
                d_out,            // d_data
                *executestream);  // stream

            // Copy output
            executestream->record(dm_job.computeEnd);
            dtohstream->waitEvent(dm_job.computeEnd);
            dtohstream->record(dm_job.outputStart);
            dtohstream->memcpyDtoHAsync(
                h_out,                         // dst
                d_out,                         // src
                ndm_gulp_current * dm_stride); // size
            dtohstream->record(dm_job.outputEnd);
        } // end for dm_job_id_inner
    } // end for dm_job_id_outer

    // Wait for final memory transfer
    dtohstream->synchronize();

    // Copy output
    dedisp_size dst_stride = nsamp_computed * out_bytes_per_sample;
    dedisp_size src_stride = nsamp_padded * out_bytes_per_sample;
    memcpy2D(
        out,        // dst
        dst_stride, // dst width
        h_data_dm,  // src
        src_stride, // src width
        dst_stride, // width bytes
        ndm);       // height
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