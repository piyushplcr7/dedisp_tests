// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include "FDDGPUPlan.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <deque>
#include <mutex>
#include <thread>

#include <assert.h>
#include "gpu_fft.hpp"
#include <omp.h>

#include "common/dedisp_strings.h"
#include "dedisperse/FDDKernel.hpp"
#include "unpack/unpack.h"
#define DEDISP_BENCHMARK
#ifdef DEDISP_BENCHMARK
#include "external/Stopwatch.h"
#endif

#include "chunk.h"
#include "common/helper.h"
#include "helper.h"
#include "fdd_gpu.h"
#include "cufft_optimal_size.hpp"
#include "fitscontainer.hpp"

#include <unistd.h>   
#include <fcntl.h>    
#include <mpi.h>

namespace dedisp {
// Constructor
FDDGPUPlan::FDDGPUPlan(size_type nchans, float_type dt, float_type f0,
                       float_type df, int device_idx)
    : GPUPlan(nchans, dt, f0, df, device_idx), device_idx_(device_idx) {}

FDDGPUPlan::FDDGPUPlan(const dataLoader& container, int device_idx)
    : GPUPlan(container.nchansLocal(), 
              container.sampletime(), 
              container.f0(),
              container.ddf(), 
              container.avgvoverc(),
              device_idx),
      device_idx_(device_idx),
      nsamps_(container.nsampsLocal()),
      dt_(container.sampletime())
               {
  container_ = &container;
#ifdef TESTDEDISP_DEBUG
  printf("dt = %f\n", container.sampletime());
  printf("f0              = %f\n", container.f0());
  printf("double ddf      = %.15f\n", container.ddf());
  printf("df              = %.15f\n", (float)container.ddf());
  printf("bw              = %f\n", container.bw());
#endif
}

// Destructor
FDDGPUPlan::~FDDGPUPlan() {}

void FDDGPUPlan::writeOutput(char* outfile, int w, bool barycenter, const std::vector<int>& inForOut) {
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

    int i = 0;
    for (i = 0 ; i < inForOut.size() ; ++i) {
      if (inForOut[i] >= nsamps_computed_) {
        Nout = i;
        break;
      }
    }
    
    #pragma omp parallel
    {
      // Create buffer for doing barycentering
      float* barycentered_data = new float[Nout];
      #pragma omp for
      for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
        char out_file_name[256];
        sprintf(out_file_name,"%s_DM%.*f.%s", outfiles_basename, w, dmlist[out_file_idx], "dat");
        
        // Write block
        float* dedispersed_data = output + out_file_idx * (size_t)nsamps_computed_;
        // Copy data while barycentering
        for (int i = 0 ; i < Nout ; ++i) {
          barycentered_data[i] = dedispersed_data[inForOut[i]];
        }

        int fd = open(out_file_name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            std::cerr << "Open failed\n";
            std::exit(1);
        }
        size_t numtowrite = (size_t)Nout * out_nbits / 8;

        ssize_t written = write(
            fd,
            barycentered_data,
            numtowrite
        );

        if (written != (ssize_t)numtowrite) {
            std::cerr << "Write failed\n";
            std::exit(1);
        }

        close(fd);
        
      }

      delete[] barycentered_data;
    }
    
  } 
  else if (multout_ && !fftout_ && !barycenter) {
    std::cout << "**********************Check for writing function" << std::endl;
    #pragma omp parallel
    {
      #pragma omp for
      for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
        char out_file_name[256];
        sprintf(out_file_name,"%s_DM%.*f.%s", outfiles_basename, w, dmlist[out_file_idx], "dat");

        int fd = open(out_file_name, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
            std::cerr << "Open failed\n";
            std::exit(1);
        }
        size_t numtowrite = (nsamps_ - get_max_delay())/2 * 2; //nsamps_ + get_max_delay(); 
        size_t bytestowrite = numtowrite * out_nbits / 8;
        size_t dm_offset = get_max_delay(); //roundf(dmlist[out_file_idx] * h_delay_table[0] );
        ssize_t written = write(
            fd,
            output + out_file_idx * (size_t)nsamps_computed_ + dm_offset,
            bytestowrite
        );

        if (written != bytestowrite) {
            std::cerr << "Write failed\n";
            std::exit(1);
        }

        close(fd);
        
      }

    }
    
  } 
  else if (multout_ && fftout_) {
    #pragma omp parallel for 
    for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
      char out_file_name[256];
      sprintf(out_file_name,"%s_DM%.*f.%s", outfiles_basename, w, dmlist[out_file_idx], "fft");
      
      // Write block
      FILE* file_out = fopen(out_file_name, "wb");
      size_t numtowrite = (size_t)nsamp_padded_ * out_nbits / 8;

      size_t writtennum = fwrite(output + out_file_idx * (size_t)nsamp_padded_, 
            1, 
            numtowrite, 
            file_out);

      if (writtennum != numtowrite) {
        std::cerr << "Writing file " << out_file_idx << " failed!" << std::endl;
      }

      fclose(file_out);
      
    }
  }
  else {
    FILE *file_out;
    if (fftout_) {
      printf("Writing the output Fourier coefficients of all DMs in one file\n");
      char out_file_name[256];
      sprintf(out_file_name,"%s.allDMs.fft", outfiles_basename);
      file_out = fopen(out_file_name, "wb");
    } else {
      printf("Writing the output dedispersed timeseries of all DMs in one file\n");
      char out_file_name[256];
      sprintf(out_file_name,"%s.allDMs.dat", outfiles_basename);
      file_out = fopen(out_file_name, "wb");
    }

    fwrite(output, 1, (size_t)(fftout_? nsamp_padded_ : nsamps_computed_) * dm_count * out_nbits / 8,
          file_out);
    fclose(file_out);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();
                    std::cout << "Writing the output took " << (double)duration_us / 1e6
                    << " seconds" << std::endl;
}

void FDDGPUPlan::writeInfs(char* outfile, const dataFile* file, size_t nsamps, double dt, int w, bool barycenter, double blotoa, double avgvoverc) {
  const char* outfiles_basename = (outfile == NULL) ? "output" : outfile;

  unsigned dm_count = m_dm_count;
  const float* dmlist = get_dm_list();

  // No infs to be written if multout is not set
  if (!multout_)
    return;

  #pragma omp parallel for 
  for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
    char out_inf_name[256];
    sprintf(out_inf_name,"%s_DM%.*f.inf", outfiles_basename, w, dmlist[out_file_idx]);

    FILE* inf_out = fopen(out_inf_name,"w");

    // Writing the inf data
    fprintf(inf_out,"%-40s=  %s_DM_%.*f\n", " Data file name without suffix", outfiles_basename, w, dmlist[out_file_idx]);
    fprintf(inf_out,"%-40s=  %s\n", " Telescope used", file->telescope());
    fprintf(inf_out,"%-40s=  %s\n", " Instrument used", file->instrument());
    fprintf(inf_out,"%-40s=  %s\n", " Object being observed", file->objectName());
    fprintf(inf_out,"%-40s=  %s\n", " J2000 Right Ascension (hh:mm:ss.ssss)", file->rightAscension());
    fprintf(inf_out,"%-40s=  %s\n", " J2000 Declination     (dd:mm:ss.ssss)", file->declination()[0] == '+' ? file->declination() + 1 : file->declination());
    fprintf(inf_out,"%-40s=  %s\n", " Data observed by", file->observer());
    double epoch;
    if (barycenter) {
      // Highest channel frequency (observed), Doppler-shifted to emitted frequency
      double baryhifreq = file->freqs().back() * (1.0 + avgvoverc);
      // Dispersion delay at highest channel: DM / (0.000241 * f_emitted^2) in seconds
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
    fprintf(inf_out,"%-40s=  0, %ld\n", " On/Off bin pair #  1 ", nsamps-1); // Check!
    fprintf(inf_out,"%-40s=  %ld, %ld\n", " On/Off bin pair #  2", nsamps-1, nsamps-1);
    fprintf(inf_out,"%-40s=  Radio\n", " Type of observation (EM band)  ");
    fprintf(inf_out,"%-40s=  900\n", " Beam diameter (arcsec)");
    fprintf(inf_out,"%-40s=  %.*f\n", " Dispersion measure (cm-3 pc)", w, dmlist[out_file_idx]);
    fprintf(inf_out,"%-40s=  %.7f\n", " Central freq of low channel (MHz)", file->freqs()[0]);
    fprintf(inf_out, "%-40s=  %.7f\n", " Total bandwidth (MHz)", file->bw());
    fprintf(inf_out, "%-40s=  %d\n", " Number of channels", file->nchan());
    fprintf(inf_out, "%-40s=  %.15f\n", " Channel bandwidth (MHz)", -file->ddf());

    char *user = getenv("USER");  // Get the username
    if (!user) user = getenv("USERNAME");  // Fallback for Windows

    fprintf(inf_out, "%-40s=  %s\n", " Data analyzed by", user ? user : "Unknown");
    fprintf(inf_out, " Any additional notes: \n \tProject ID %s, Date: %s.\n \t4 polns were not summed.  Samples have 8 bits. \n", file->projid(), file->dateobs());
    
    fclose(inf_out);
    
  }
}

// Ensure this function is called after generating the DM list
void FDDGPUPlan::setOutputParams(
        bool cleanout,
        bool fftout,
        bool multout,
        int out_nbits,
        char* outfile,
        int w,
        bool barycenter) {
  //
  cleanout_ = cleanout;
  fftout_ = fftout;
  multout_ = multout;
  outfile_ = outfile;
  w_ = w;
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
  
  // nsamp_fft is the fft size. Keeping this bigger than nsamps is implicitly
  // adding zero padding to the end of the timeseries. Choosing nsamps + max_delay
  // as nsamps_fft prevents contamination at the ends when combined with shifts to 
  // right. The time samples even when missing info from channels remain chronologically
  // relevant.

  // nsamp_padded is chosen large enough to hold complex coefficients arising 
  // from fourier transforms. 

  nsamp_fft_ = closestOptimal(nsamps_ + max_delay,true);
  nsamp_padded_ = 2ULL * (nsamp_fft_/2 + 1);

  // output clean timeseries
  if (cleanout_) {
      nsamps_computed_ = nsamps_ - max_delay;
  }
  // output dirty timeseries
  else {
      std::cout << "---------------nsamps_computed_ = nsamps_ + max_delay" << std::endl;
      std::cout << "nsamps_ = " << nsamps_ << std::endl;
      std::cout << "max_delay = " << max_delay << std::endl;
      nsamps_computed_ = nsamps_ + max_delay;
  }

  // Make nsamps_computed even
  //nsamps_computed_ = (nsamps_computed_/2) * 2;

  if (fftout) {
    printf("Computing %lu Fourier Coefficients of dedispersed timeseries "
          "(adjusting for max delay)\n",
          nsamp_fft_);
    printf("Output data array size : %lu MB\n",
          (dm_count * nsamp_fft_ * sizeof(float)) / (1 << 20));

    // Output is chosen such that it is able to hold all the FFT coefficients
    output_buffer_ = std::make_unique<float[]>(nsamp_padded_ * dm_count);
  }
  else {
    printf("Computing %lu out of %lu total samples (%.2f%% efficiency)\n",
         nsamps_computed_, nsamps_,
         100.0 * (dedisp_float)nsamps_computed_ / nsamps_);
    printf("Output data array size : %lu MB\n",
          (dm_count * nsamps_computed_ * sizeof(float)) / (1 << 20));
    
    output_buffer_ = std::make_unique<float[]>(nsamps_computed_ * dm_count);
  }

  if (output_buffer_ == nullptr) {
    printf("\nERROR: Failed to allocate output array\n");
    exit(-1);
  }

}

// Public interface for FDD on GPU
void FDDGPUPlan::execute(size_type nsamps, const byte_type *in,
                         size_type in_nbits, byte_type *out,
                         size_type out_nbits, unsigned flags) {
  char *use_segmented_str = getenv("USE_SEGMENTED");
  bool use_segmented = !use_segmented_str ? false : atoi(use_segmented_str);
  if (use_segmented || true) {
#ifdef TESTDEDISP_DEBUG
    std::cout << ">> Running segmented GPU implementation (per-file segments)" << std::endl;
#endif
    if (container_ == nullptr) {
      std::cerr << "USE_SEGMENTED requires FDDGPUPlan constructed from a dataLoader." << std::endl;
      exit(-1);
    }
    if (!container_->segmentsEqualLength()) {
      std::cerr << "USE_SEGMENTED requires all per-file segments to have equal length." << std::endl;
      exit(-1);
    }

    const int    nseg               = 1; //container_->numLocFits();
    const size_t nsamps_segment     = container_->nsampsLocal() ;//nsampsPerSegment();
    const size_t out_seg_byte_width = nsamps_segment * (out_nbits / 8);

    for (int i = 0; i < nseg; ++i) {
      const byte_type* seg_in  = container_->getSegmentPtr(i);
      byte_type*       seg_out = out + i * out_seg_byte_width;
      execute_gpu(nsamps_segment, seg_in, in_nbits,
                                seg_out, out_nbits);
    }
  } else { // Default
#ifdef DEDISP_DEBUG
    std::cout << ">> Running GPU implementation" << std::endl;
#endif
    execute_gpu(nsamps, in, in_nbits, out, out_nbits);
  }
}

// Private interface for FDD on GPU
void FDDGPUPlan::execute_gpu(size_type nsamps, const byte_type *in,
                             size_type in_nbits, byte_type *out,
                             size_type out_nbits) {
  enum {
    BITS_PER_BYTE = 8,
    BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
  };

  //aa_gpu_timer C2Rtimer;
  double c2rtime = 0;

  //aa_gpu_timer R2Ctimer;
  double r2ctime = 0;

  // Original code. Commented out to allow working directly with floats
  if (in_nbits == 8) {
    std::cout << "****************************************************\n";
    std::cout << " Using 8 bit input. Unpacking to 32 bit floats on GPU.\n";
    std::cout << "****************************************************\n";
  }
  else if (in_nbits == 32) {
    std::cout << "****************************************************\n";
    std::cout << " Using 32 bit float input. No unpacking needed.\n";
    std::cout << "****************************************************\n";
  }
  else {
    std::cerr << "Unsupported input bit depth: " << in_nbits << std::endl;
    exit(1);
  }

  assert(out_nbits == 32);

  // Parameters
  float dt = m_dt;                      // sample time
  unsigned int nchan = m_nchans;        // number of observering frequencies
  size_t nsamp = nsamps;          // number of time samples
  unsigned int ndm = m_dm_count;        // number of DMs

  // Need to modify nsamp_fft_ and nsamp_padded_ for segment dedispersion
  dedisp_size max_delay = get_max_delay();
  size_t nsamp_fft_segment = closestOptimal(nsamp + max_delay,true);
  size_t nsamp_padded_segment = 2ULL * (nsamp_fft_segment/2 + 1);
  size_t nfreq = (nsamp_fft_segment / 2 + 1);    // number of spin frequencies
  //size_t nsamps_computed_segment = nsamp + max_delay;

  std::cout << "nsamp_segment        = " << nsamp << std::endl;
  std::cout << "nsamp_fft_segment    = " << nsamp_fft_segment << std::endl;
  std::cout << "nsamp_padded_segment = " << nsamp_padded_segment << std::endl;

  std::cout << "m_max_delay = " << m_max_delay << std::endl;
  std::cout << "nsamps_computed = " << nsamps_computed_ << std::endl;
#ifdef DEDISP_DEBUG
  std::cout << debug_str << std::endl;
  std::cout << "nsamp_fft    = " << nsamp_fft_ << std::endl;
  std::cout << "nsamp_padded = " << nsamp_padded_ << std::endl;
#endif

  // Maximum number of DMs computed in one gulp
  // Parameters might be tuned for efficiency depending on system architecture
  unsigned int ndm_batch_max = std::min(ndm / 4, (unsigned int)64);
  unsigned int ndm_fft_batch = 32;
  ndm_fft_batch = std::min(ndm_batch_max, ndm_fft_batch);
  // The number of buffers for DM results is configured below based on the
  // amount of available GPU memory.
  unsigned int ndm_buffers = 1;

  // Maximum number of channels processed in one gulp
  // Parameters might be tuned for efficiency depending on system architecture
  unsigned int nchan_batch_max = std::min(nchan / 4, (unsigned int)64);
  unsigned int nchan_fft_batch = 64;
  unsigned int nchan_buffers = 2;

  // Verbose iteration reporting
#ifdef DEDISP_DEBUG
  bool enable_verbose_iteration_reporting = false;
#endif

  // Compute derived counts
  dedisp_size out_bytes_per_sample =
      out_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);
  
  dedisp_size chans_per_word = sizeof(dedisp_word) * BITS_PER_BYTE / in_nbits;
  //dedisp_size chans_per_word = 1;

  // The number of channel words in the input
  dedisp_size nchan_words = nchan / chans_per_word;

  // The number of channel words proccessed in one gulp
  dedisp_size nchan_words_gulp = nchan_batch_max / chans_per_word;

  if (nchan_batch_max % chans_per_word != 0) {
    std::cerr << "nchan_batch_max must be a multiple of chans_per_word" << std::endl;
    exit(1);
  }

  // Events, markers, timers
  cu::Event eStartGPU, eEndGPU;
  cu::Marker mAllocMem("Allocate host and device memory", cu::Marker::black);
  cu::Marker mCopyMem("Copy CUDA mem to CPU mem", cu::Marker::black);
  cu::Marker mMPI("MPI Communication and file write thread", cu::Marker::red);
  cu::Marker mMPI1("Packing into sendbuf", cu::Marker::blue);
  cu::Marker mMPI2("Unpacking-memadd2d to output", cu::Marker::green);
  cu::Marker mPrepFFT("cufft Plan Many", cu::Marker::yellow);
  cu::Marker mPrepSpinf("spin Frequency generation", cu::Marker::blue);
  cu::Marker mDelayTable("Delay table copy", cu::Marker::black);
  cu::Marker mExeGPU("Dedisp fdd execution on GPU", cu::Marker::green);
#ifdef DEDISP_BENCHMARK
  std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> input_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> gpuexec_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
  total_timer->Start();
  init_timer->Start();
#endif

  if ((nsamp_fft_segment - (int) nsamp_fft_segment) != 0 || (nsamp_padded_segment - (int) nsamp_padded_segment) != 0) {
    std::cerr << "Too large sizes for nsamp_fft or nsamp_padded. Check FDDGPUPlan.cpp" << std::endl;
    exit(1);
  }

  // Prepare cuFFT plans
#ifdef DEDISP_DEBUG
  std::cout << fft_plan_str << std::endl;
#endif
  mPrepFFT.start();
  gpufftHandle plan_r2c, plan_c2r;
  int n[] = {(int)nsamp_fft_segment};
  int rnembed[] = {(int)nsamp_padded_segment};     // width in real elements
  int cnembed[] = {(int)nsamp_padded_segment / 2}; // width in complex elements

  gpufftResult result =
      gpufftPlanMany(&plan_r2c,              // plan
                    1, n,                   // rank, n
                    rnembed, 1, rnembed[0], // inembed, istride, idist
                    cnembed, 1, cnembed[0], // onembed, ostride, odist
                    GPUFFT_R2C,              // type
                    nchan_fft_batch);       // batch
  if (result != GPUFFT_SUCCESS) {
    throw std::runtime_error("Error creating real to complex FFT plan.");
  }
  gpufftSetStream(plan_r2c, *executestream);

  result =
      gpufftPlanMany(&plan_c2r,              // plan
                    1, n,                   // rank, n
                    cnembed, 1, cnembed[0], // inembed, istride, idist
                    rnembed, 1, rnembed[0], // onembed, ostride, odist
                    GPUFFT_C2R,              // type
                    ndm_fft_batch);         // batch
  if (result != GPUFFT_SUCCESS) {
    throw std::runtime_error("Error creating complex to real FFT plan.");
  }
  gpufftSetStream(plan_c2r, *executestream);
  mPrepFFT.end();

  // Generate spin frequency table
  mPrepSpinf.start();
  if (h_spin_frequencies.size() != nfreq) {
    generate_spin_frequency_table(nfreq, nsamp_fft_segment, dt);
  }
  mPrepSpinf.end();

  // Determine the amount of memory to use
  size_t d_memory_total = m_device->get_total_memory();
  size_t d_memory_free = m_device->get_free_memory();
  size_t sizeof_data_t_nu =
      1ULL * nsamp * nchan_words_gulp * sizeof(dedisp_word);
  size_t sizeof_data_x_nu =
      1ULL * nchan_batch_max * nsamp_padded_segment * sizeof(float);
  size_t sizeof_data_x_dm = 1ULL * ndm_batch_max * nsamp_padded_segment * sizeof(float);
  // For device side, initial value
  size_t d_memory_required = sizeof_data_t_nu * nchan_buffers +
                             sizeof_data_x_nu * 1 +
                             sizeof_data_x_dm * ndm_buffers;
  size_t d_memory_reserved = 0.05 * d_memory_total;

  // Subtract the memory usage of any pre-existing device buffers
  size_t d_memory_in_use = 0;
  for (cu::DeviceMemory &d_memory : d_data_t_nu_) {
    d_memory_in_use += d_memory.size();
  }
  for (cu::DeviceMemory &d_memory : d_data_x_dm_) {
    d_memory_in_use += d_memory.size();
  }
  d_memory_free += d_memory_in_use;

  // Iteratively search for a maximum amount of ndm_buffers, with safety margin
  // Make sure that it fits on device memory
  while ((ndm_buffers * ndm_batch_max) < ndm &&
         (d_memory_required + d_memory_reserved + sizeof_data_x_dm) <
             d_memory_free) {
    ndm_buffers++;
    d_memory_required = sizeof_data_t_nu * nchan_buffers +
                        sizeof_data_x_nu * 1 + sizeof_data_x_dm * (ndm_buffers);
  };

  // Debug
//#ifdef DEDISP_DEBUG
  std::cout << debug_str << std::endl;
  std::cout << "ndm_buffers     = " << ndm_buffers << " x " << ndm_batch_max
            << " DMs" << std::endl;
  /* std::cout << "nchan_buffers   = " << nchan_buffers << " x " << nchan_batch_max
            << " channels" << std::endl; */
  std::cout << "Device memory total    = " << d_memory_total / std::pow(1024, 3)
            << " Gb" << std::endl;
  std::cout << "Device memory free     = " << d_memory_free / std::pow(1024, 3)
            << " Gb" << std::endl;
  std::cout << "Device Memory required = "
            << d_memory_required / std::pow(1024, 3) << " Gb" << std::endl;
  std::cout << "Host memory total    = "
            << get_total_memory() / std::pow(1024, 1) << " Gb" << std::endl;
  std::cout << "Host memory free     = "
            << get_free_memory() / std::pow(1024, 1) << " Gb" << std::endl;
//#endif

  // Allocate memory
#ifdef DEDISP_DEBUG
  std::cout << memory_alloc_str << std::endl;
#endif
  mAllocMem.start();
  /*
      The buffers are used as follows:
      1) copy into page-locked buffer: in -> memcpyHtoH -> h_data_t_nu
      2) copy to device: h_data_t_nu -> memcopyHtoD -> d_data_t_nu
      3) unpack and transpose: d_data_t_nu -> transpose_unpack -> d_data_x_nu
      4) in-place Fourier transform: d_data_x_nu -> fft_r2c -> d_data_x_nu
      5) apply dedispersion: d_data_x_nu -> dedispserse -> d_data_x_dm
      6) in-place Fourier transform: d_data_x_dm -> fft_c2r -> d_data_x_dm
      7) copy to host: d_data_x_dm -> memcpyDtoH -> h_data_t_dm

      The suffixes have the following meaning:
      * The _t indicates that the buffer contains time domain data
      * The _f indicates that the buffer contains Fourier domain data
      * The _x indicates that the type of data various throughout processing
      * The _nu indicates input data with observing frequencies as outer
     dimension
      * The _dm indicates output data with DM as outer dimension

      The vectors (with _ suffix) are used to implement multiple-buffering
  */
  h_data_t_nu_.resize(nchan_buffers);
  h_data_t_dm_.resize(ndm_buffers);
  d_data_t_nu_.resize(nchan_buffers);
  d_data_x_dm_.resize(ndm_buffers);
  cu::DeviceMemory d_data_x_nu(sizeof_data_x_nu);
  for (unsigned int i = 0; i < nchan_buffers; i++) {
    h_data_t_nu_[i].resize(sizeof_data_t_nu);
    d_data_t_nu_[i].resize(sizeof_data_t_nu);
  }
  for (unsigned int i = 0; i < ndm_buffers; i++) {
    h_data_t_dm_[i].resize(sizeof_data_x_dm);
    d_data_x_dm_[i].resize(sizeof_data_x_dm);
  }
  mAllocMem.end();

  MPI_Barrier(MPI_COMM_WORLD);

#ifdef DEDISP_DEBUG
  size_t d_memory_free_after_malloc = m_device->get_free_memory(); // bytes
  size_t h_memory_free_after_malloc = get_free_memory();           // MB
  std::cout << "Device memory free after memory allocations    = "
            << d_memory_free_after_malloc / std::pow(1024, 3) << " Gb"
            << std::endl;
  std::cout << "Host memory free after memory allocations    = "
            << h_memory_free_after_malloc / std::pow(1024, 1) << " Gb"
            << std::endl;
#endif

  // Initialize FDDKernel
  FDDKernel kernel;
  mDelayTable.start();
  kernel.copy_delay_table(d_delay_table, m_nchans * sizeof(dedisp_float), 0,
                          *htodstream);
  mDelayTable.end();
#ifdef DEDISP_BENCHMARK
  init_timer->Pause();
#endif

  struct ChannelData {
    unsigned int ichan_start;
    unsigned int ichan_end;
    unsigned int nchan_current;
    void *h_in_ptr;
    void *d_in_ptr;
    cu::Event inputStart, inputEnd;
    cu::Event preprocessingStart, preprocessingEnd;
    cu::Event outputStart, outputEnd;
  };

  // Configure ChannelData jobs
  unsigned int nchan_jobs = (nchan + nchan_batch_max) / nchan_batch_max;
  std::vector<ChannelData> channel_jobs(nchan_jobs);
  for (unsigned job_id = 0; job_id < nchan_jobs; job_id++) {
    ChannelData &job = channel_jobs[job_id];
    job.ichan_start = job_id == 0 ? 0 : channel_jobs[job_id - 1].ichan_end;
    job.nchan_current = std::min(nchan_batch_max, nchan - job.ichan_start);
    job.ichan_end = job.ichan_start + job.nchan_current;
    job.h_in_ptr = h_data_t_nu_[job_id % nchan_buffers];
    job.d_in_ptr = d_data_t_nu_[job_id % nchan_buffers];
    if (job.nchan_current == 0) {
      channel_jobs.pop_back();
    }
  }

  struct DMData {
    unsigned int idm_start;
    unsigned int idm_end;
    unsigned int ndm_current;
    std::mutex cpu_lock;
    std::mutex gpu_lock;
    std::mutex out_lock;
    cu::HostMemory *h_data_t_dm;
    cu::DeviceMemory *d_data_x_dm;
    cu::Event inputStart, inputEnd;
    cu::Event dedispersionStart, dedispersionEnd;
    cu::Event postprocessingStart, postprocessingEnd;
    cu::Event outputStart, outputEnd;
  };

  // Configure DMData jobs
  unsigned int ndm_jobs = (ndm + ndm_batch_max) / ndm_batch_max;
  std::vector<DMData> dm_jobs(ndm_jobs);
  for (unsigned job_id = 0; job_id < ndm_jobs; job_id++) {
    DMData &job = dm_jobs[job_id];
    job.idm_start = job_id == 0 ? 0 : dm_jobs[job_id - 1].idm_end;
    job.ndm_current = std::min(ndm_batch_max, ndm - job.idm_start);
    job.idm_end = job.idm_start + job.ndm_current;
    job.h_data_t_dm = &h_data_t_dm_[job_id % ndm_buffers];
    job.d_data_x_dm = &d_data_x_dm_[job_id % ndm_buffers];
    if (job.ndm_current == 0) {
      dm_jobs.pop_back();
    }
    job.cpu_lock.lock();
    job.out_lock.lock();
    if (job_id >= ndm_buffers) {
      job.gpu_lock.lock();
    }
  }

  // Launch thread to manage MPI communication, data reduction and output
  std::thread mpi_thread = std::thread([&]() {
    mMPI.start();
    // Buffers for communication
    float* sendbuf = new float[max_delay * ndm_batch_max];
    float* recvbuf = new float[max_delay * ndm_batch_max];

    int mpi_rank = container_->getMPIRank();
    int mpi_size = container_->getMPISize();

    // If last rank, no destination, otherwise the next rank is the dest
    int dest = (mpi_rank == mpi_size - 1) ? MPI_PROC_NULL : mpi_rank + 1;

    // If first rank, no source, otherwise the previous rank is source
    int source = (mpi_rank == 0) ? MPI_PROC_NULL : mpi_rank - 1;

    // Bounded ring of in-flight per-DM file writes. Each entry owns one
    // MPI_File handle and its outstanding MPI_File_iwrite_at request. We
    // close (which waits) once the ring is full or at the end of the run.
    // Sized small enough to stay clear of common fd limits even with
    // hundreds of ranks * tens of DMs in flight per rank.
    // TODO: source outfile_basename and dmstepW from a setter (or extend
    // setOutputParams). Hardcoded defaults below mirror writeOutput().
    // TODO: nsamps > INT_MAX / 4 → derived datatype or MPI 4.0 _c API.
    struct PendingWrite { 
      MPI_File fh; 
      MPI_Request req; 
    };

    constexpr size_t max_inflight_writes = 32;
    
    std::deque<PendingWrite> inflight_writes;
    
    auto drain_one_write = [&]() {
      auto &pw = inflight_writes.front();
      MPI_Wait(&pw.req, MPI_STATUS_IGNORE);
      MPI_File_close(&pw.fh);
      inflight_writes.pop_front();
    };

    //const char* outfile_ = "/scratch/panchal/segmentout/oot";
    //const int   w_          = 2;
    const dedisp_float* dmlist   = get_dm_list();

    // Per-rank slice within each DM row of the paged `out` buffer:
    //   rank 0   : skip first max_delay junk samples, write nsamps - max_delay
    //   rank r>0 : skip nothing, write nsamps samples
    // Global byte offset places rank r's slice immediately after rank r-1.
    const size_t local_skip_bytes = (mpi_rank == 0)
        ? (size_t)max_delay * out_bytes_per_sample : 0;

    const size_t write_bytes      = (mpi_rank == 0)
        ? (size_t)(nsamps - max_delay) * out_bytes_per_sample
        : (size_t)nsamps * out_bytes_per_sample;

    const MPI_Offset global_byte_off = (mpi_rank == 0)
        ? (MPI_Offset)0
        : (MPI_Offset)((size_t)mpi_rank * nsamps - max_delay)
            * out_bytes_per_sample;

    const int mpi_count = (int)(write_bytes / sizeof(float));

    // For barycentering, the dedispersed data is the input. The input is
    // split across multiple procs. 
    float* barycentered_data;

    int global_input_idx_start,
        global_input_idx_end,
        global_output_idx_start,
        global_output_idx_end,
        local_Nout;

    MPI_Offset global_byte_off_barycenter = 0;

    // Global resample map (barycentric sample positions) -- the same source the
    // driver passes to writeOutput(). Safe to fetch unconditionally: it is empty
    // under -nobary and the barycenter_ blocks below are skipped in that case.
    const std::vector<int>& inForOut = container_->getInForOut();

    if (barycenter_) {
      global_input_idx_start = (mpi_rank == 0) ? 0 : mpi_rank * nsamps - max_delay;

      global_input_idx_end = global_input_idx_start
                           + (mpi_rank > 0 ? nsamps : nsamps - max_delay);

      // Go over the global inForOut to find important positions
      int out_idx_global;
      for (out_idx_global = 0 ; out_idx_global < (int)inForOut.size() ; ++out_idx_global) {
        if (inForOut[out_idx_global] >= global_input_idx_start)
          break;
      } 

      // global_output_idx_start contains the position in the output, from where
      // the contributions by the current MPI proc start.  
      global_output_idx_start = out_idx_global;

      for (; out_idx_global < inForOut.size(); ++out_idx_global) {
        if (inForOut[out_idx_global] >= global_input_idx_end)
          break;
      }

      // global_output_idx_end contains position in the output, from where
      // the contributions by another MPI proc start.
      global_output_idx_end = out_idx_global;
      local_Nout = global_output_idx_end - global_output_idx_start;

      barycentered_data = new float[(size_t)local_Nout * ndm_batch_max];
    }

    // Go over all DM jobs
    for (unsigned job_id = 0; job_id < dm_jobs.size(); job_id++) {
      auto &dm_job = dm_jobs[job_id];

      dedisp_size out_stride = 1ULL * nsamps_computed_ * out_bytes_per_sample;
      dedisp_size out_offset = 1ULL * dm_job.idm_start * out_stride;
      auto* out_curr_dmjob = (void*)out + out_offset;
      auto *out_curr_dmjob_R = out_curr_dmjob + 1ULL * nsamps * out_bytes_per_sample;

      // Try to acquire the out_lock. It can be done only when the output buffer is populated
      dm_job.out_lock.lock();
      mMPI1.start();
      // Perform MPI communication for current DM job
      // We first pack the R data into the contiguous buffer sendbuf
      if (mpi_rank != mpi_size - 1) {
        memcpy2D((void*) sendbuf,           // dst
                max_delay * out_bytes_per_sample,          // dst stride
                out_curr_dmjob_R,           // src
                out_stride,          // src stride
                max_delay * out_bytes_per_sample,           // width bytes
                dm_job.ndm_current); // height
      }

      mMPI1.end();

      // Send recv call for communication
      MPI_Sendrecv(
          sendbuf, max_delay * ndm_batch_max, MPI_FLOAT, dest,   mpi_rank, // mpi_rank_ sends this message
          recvbuf, max_delay * ndm_batch_max, MPI_FLOAT, source, mpi_rank - 1,
          MPI_COMM_WORLD, MPI_STATUS_IGNORE
      );

      mMPI2.start();
      // Unpacking the data and doing memadd2D if rank not first
      if (mpi_rank != 0) {
        memadd2D(out_curr_dmjob,             // dst
                out_stride,          // dst stride
                (void*) recvbuf,             // src
                max_delay * out_bytes_per_sample,          // src stride
                max_delay * out_bytes_per_sample,            // width bytes
                dm_job.ndm_current); // height
      }
      mMPI2.end();

      // Output buffer ready with dedispersed data. Perform barycentering if needed
      if (barycenter_) {
        // Resample loop
        for (unsigned j = 0 ; j < dm_job.ndm_current; ++j) {
          // Do the resample
          float* barycentered_data_row_j = barycentered_data + local_Nout * j;
          float* dedispersed_data_row_j = (float*)(out_curr_dmjob + (size_t)j * out_stride + local_skip_bytes);

          for (int I = global_output_idx_start ; I < global_output_idx_end ; ++I) {
            barycentered_data_row_j[I-global_output_idx_start] = 
              dedispersed_data_row_j[inForOut[I]-global_input_idx_start];
          }
        }

        // MPI write loop
        for (unsigned j = 0 ; j < dm_job.ndm_current; ++j) {
          float* barycentered_data_row_j = barycentered_data + local_Nout * j;

          // Preparing the file name
          const unsigned idm = dm_job.idm_start + j;
          char outname[256];
          snprintf(outname, sizeof(outname), "%s_DM%.*f.dat",
                  outfile_, w_, dmlist[idm]);

          // TODO: Use synchronous MPI write call
          MPI_File fh; 
          MPI_File_open(MPI_COMM_SELF, outname,
                        MPI_MODE_CREATE | MPI_MODE_WRONLY,
                        MPI_INFO_NULL, &fh);

          MPI_File_write_at(fh, global_output_idx_start,
                            barycentered_data_row_j,
                            local_Nout, MPI_FLOAT, MPI_STATUS_IGNORE);

          MPI_File_close(&fh);
        }

      }
      else {
        // Output buffer for current DM batch is ready. Launch non-blocking
        // per-DM writes (one file per DM, MPI_COMM_SELF so ranks proceed
        // independently). Disjoint byte ranges per rank → no shared-file
        // consistency concern. Ring-drain throttles outstanding writes.
        for (unsigned j = 0; j < dm_job.ndm_current; ++j) {
          if (inflight_writes.size() >= max_inflight_writes) {
            drain_one_write();
          }

          const unsigned idm = dm_job.idm_start + j;
          char outname[256];
          snprintf(outname, sizeof(outname), "%s_DM%.*f.dat",
                  outfile_, w_, dmlist[idm]);

          PendingWrite pw;
          MPI_File_open(MPI_COMM_SELF, outname,
                        MPI_MODE_CREATE | MPI_MODE_WRONLY,
                        MPI_INFO_NULL, &pw.fh);

          const char* row_base =
              (const char*)out_curr_dmjob + (size_t)j * out_stride;
          MPI_File_iwrite_at(pw.fh, global_byte_off,
                            row_base + local_skip_bytes,
                            mpi_count, MPI_FLOAT, &pw.req);

          inflight_writes.push_back(pw);
        }
      }
      
    } // Loop over dm jobs

    if (barycenter_) {
      delete[] barycentered_data;
    }
    else {
      // Drain any remaining in-flight writes before exiting the thread.
      while (!inflight_writes.empty()) {
        drain_one_write();
      }
    }

    // De-allocate the buffers
    delete[] sendbuf;
    delete[] recvbuf;
    mMPI.end();
  });

  // Launch thread to copy output data from device to host for each dm_job
  std::thread output_thread = std::thread([&]() {
    gpuSetDevice(device_idx_);

    for (unsigned job_id = 0; job_id < dm_jobs.size(); job_id++) {
      auto &dm_job = dm_jobs[job_id];

      // Wait for DtoH copy to finish for this job
      dm_job.cpu_lock.lock();
      dm_job.outputEnd.synchronize();

      // Info
#ifdef DEDISP_DEBUG
      if (enable_verbose_iteration_reporting) {
        std::cout << "Copy output " << dm_job.idm_start << " to "
                  << dm_job.idm_end << " with " << dm_job.ndm_current << " ndms"
                  << std::endl;
      }
#endif
      // copy part from pinned h_data_t_dm to part of paged return buffer out
      // GPU Host mem pointers
      dedisp_size src_stride = 1ULL * nsamp_padded_segment * out_bytes_per_sample;
      float* h_src_float = (float*)dm_job.h_data_t_dm->data();
      auto *h_src = (void*) h_src_float;

      dedisp_size dst_stride = 1ULL * nsamps_computed_ * out_bytes_per_sample;
      dedisp_size dst_offset = 1ULL * dm_job.idm_start * dst_stride;
      auto* h_dst = (void*)out + dst_offset;

      mCopyMem.start();
      // Flush the pinned buffer 
      memcpy2D(h_dst,               // dst
               dst_stride,          // dst stride
               h_src,               // src
               src_stride,          // src stride
               dst_stride,          // width bytes
               dm_job.ndm_current); // height
      mCopyMem.end();

      // Signal that the host buffer can be used again
      if ((job_id + ndm_buffers) < ndm_jobs) {
        dm_jobs[job_id + ndm_buffers].gpu_lock.unlock();
      }

      dm_job.out_lock.unlock();
    }
  });

#ifdef DEDISP_DEBUG
  std::cout << fdd_dedispersion_str << std::endl;
#endif
  htodstream->record(eStartGPU);
#ifdef DEDISP_DEBUG
  std::cout << "Finished htodstream record" << std::endl;
#endif
  mExeGPU.start();
#ifdef DEDISP_DEBUG
  std::cout << "Finished mExeGPU.start()" << std::endl;
#endif

  // Process all dm batches (outer dm jobs)
  for (unsigned dm_job_id_outer = 0; dm_job_id_outer < dm_jobs.size();
       dm_job_id_outer += ndm_buffers) {
        std::cout << "dm_job_id_outer = " << dm_job_id_outer << std::endl;
    // Process all channel batches
    for (unsigned channel_job_id = 0; channel_job_id < channel_jobs.size();
         channel_job_id++) {
      auto &channel_job = channel_jobs[channel_job_id];
#ifdef DEDISP_DEBUG
      // Info
      std::cout << "Processing channel " << channel_job.ichan_start << " to "
                << channel_job.ichan_end << std::endl;

#endif
      // Channel input size
      dedisp_size dst_stride = nchan_words_gulp * sizeof(dedisp_word);
      dedisp_size src_stride = nchan_words * sizeof(dedisp_word);

      // Copy the input data for the first job
      if (channel_job_id == 0) {
        dedisp_size gulp_chan_byte_idx =
            (channel_job.ichan_start / chans_per_word) * sizeof(dedisp_word);

        memcpy2D(channel_job.h_in_ptr,    // dst
                 dst_stride,              // dst width
                 in + gulp_chan_byte_idx, // src
                 src_stride,              // src width
                 dst_stride,              // width bytes (represents how many columns actually copied?)
                 nsamp);                  // height

        htodstream->record(channel_job.inputStart);
        htodstream->memcpyHtoDAsync(channel_job.d_in_ptr, // dst
                                    channel_job.h_in_ptr, // src
                                    nsamp * nchan_words_gulp * sizeof(dedisp_float));  // size
        htodstream->record(channel_job.inputEnd);
      }
      executestream->waitEvent(channel_job.inputEnd);

      // Modified transpose_unpack kernel to just transpose floats
      executestream->record(channel_job.preprocessingStart);

      transpose_unpack((float *)channel_job.d_in_ptr, // d_in
                       nchan_words_gulp,                    // input width
                       nsamp,                               // input height
                       nchan_words_gulp,                    // in_stride
                       nsamp_padded_segment,                        // out_stride
                       (float*)d_data_x_nu + max_delay,                         // d_out
                       in_nbits, 32,    // in_nbits, out_nbits
                       1.0 / nchan,     // scale
                       *executestream); // stream

      // Apply zero padding R
      auto dst_ptr_R = ((float *)d_data_x_nu.data()) + nsamp + max_delay;
      unsigned int nsamp_padding = nsamp_padded_segment - nsamp - max_delay;
      gpuMemset2DAsync(dst_ptr_R,                       // devPtr
                        nsamp_padded_segment * sizeof(float),  // pitch
                        0,                             // value
                        nsamp_padding * sizeof(float), // width
                        nchan_batch_max,               // height
                        *executestream);
      // Apply zero padding L
      auto dst_ptr_L = ((float *)d_data_x_nu.data());
      gpuMemset2DAsync(dst_ptr_L,                       // devPtr
                        nsamp_padded_segment * sizeof(float),  // pitch
                        0,                             // value
                        max_delay * sizeof(float), // width
                        nchan_batch_max,               // height
                        *executestream);
      
      // FFT data (real to complex) along time axis
      for (unsigned int i = 0; i < nchan_batch_max / nchan_fft_batch; i++) {
        gpufftReal *idata = (gpufftReal *)d_data_x_nu.data() +
                           i * nsamp_padded_segment * nchan_fft_batch;
        gpufftComplex *odata = (gpufftComplex *)idata;
        //R2Ctimer.Start();
        gpufftExecR2C(plan_r2c, idata, odata);
        //R2Ctimer.Stop();
        //r2ctime += R2Ctimer.Elapsed();
      }
      executestream->record(channel_job.preprocessingEnd);

      // Process DM batches (inner dm jobs)
      for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers;
           dm_job_id_inner++) {
        unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
        if (dm_job_id >= dm_jobs.size()) {
          break;
        }
        auto &dm_job = dm_jobs[dm_job_id];
#ifdef DEDISP_DEBUG
        // Info
        std::cout << "Processing DM " << dm_job.idm_start << " to "
                  << dm_job.idm_end << std::endl;

#endif
        // Initialize output to zero
        if (channel_job_id == 0) {
          // Wait for previous output copy to finish
          if (dm_job_id_outer > 0) {
            auto &dm_job_previous = dm_jobs[dm_job_id - ndm_buffers];
            dm_job_previous.outputEnd.synchronize();
          }

          dm_job.d_data_x_dm->zero(*executestream);
        }

        // Wait for temporary output from previous job to be copied
        if (channel_job_id > (nchan_buffers - 1)) {
          auto &job_previous = channel_jobs[channel_job_id - nchan_buffers];
          job_previous.outputEnd.synchronize();
        }

        // Dedispersion in frequency domain
        executestream->record(dm_job.dedispersionStart);
        auto d_out = (dedisp_float2 *)dm_job.d_data_x_dm->data();
        kernel.launch(dm_job.ndm_current,        // ndm
                      nfreq,                     // nfreq
                      channel_job.nchan_current, // nchan
                      dt,                        // dt
                      d_spin_frequencies,        // d_spin_frequencies
                      d_dm_list,                 // d_dm_list
                      d_data_x_nu,               // d_in
                      d_out,                     // d_out
                      nsamp_padded_segment / 2,  // in stride
                      nsamp_padded_segment / 2,  // out stride
                      dm_job.idm_start,          // idm_start
                      dm_job.idm_end,            // idm_end
                      channel_job.ichan_start,   // ichan_start
                      *executestream);           // stream
        executestream->record(dm_job.dedispersionEnd);
      } // end for dm_job_id_inner

      // Copy the input data for the next job (if any)
      unsigned channel_job_id_next = channel_job_id + 1;
      if (channel_job_id_next < channel_jobs.size()) {
        auto &channel_job_next = channel_jobs[channel_job_id_next];
        dedisp_size gulp_chan_byte_idx =
            (channel_job_next.ichan_start / chans_per_word) *
            sizeof(dedisp_word);

        memcpy2D(channel_job_next.h_in_ptr, // dst
                 dst_stride,                // dst width
                 in + gulp_chan_byte_idx,   // src
                 src_stride,                // src width
                 dst_stride,                // width bytes
                 nsamp);                    // height

        htodstream->record(channel_job_next.inputStart);
        htodstream->memcpyHtoDAsync(channel_job_next.d_in_ptr, // dst
                                    channel_job_next.h_in_ptr, // src
                                    nsamp * nchan_words_gulp * sizeof(dedisp_float));       // size
        htodstream->record(channel_job_next.inputEnd);
      }

      // Wait for current batch to finish
      executestream->synchronize();

      // Add input and preprocessing time for the current channel job
#ifdef DEDISP_BENCHMARK
      input_timer->Add(
          channel_job.inputEnd.elapsedTime(channel_job.inputStart));
      preprocessing_timer->Add(channel_job.preprocessingEnd.elapsedTime(
          channel_job.preprocessingStart));
#endif

      // Add dedispersion time for current dm jobs
#ifdef DEDISP_BENCHMARK
      for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers;
           dm_job_id_inner++) {
        unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
        if (dm_job_id >= dm_jobs.size()) {
          break;
        }
        auto &dm_job = dm_jobs[dm_job_id];

        dedispersion_timer->Add(
            dm_job.dedispersionEnd.elapsedTime(dm_job.dedispersionStart));
      }
#endif
    } // end for ichan_start

    // Output DM batches
    for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers;
         dm_job_id_inner++) {
      unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
      if (dm_job_id >= dm_jobs.size()) {
        break;
      }
      auto &dm_job = dm_jobs[dm_job_id];
#ifdef DEDISP_DEBUG
      // Info

      std::cout << "Post-processing DM " << dm_job.idm_start << " to "
                << dm_job.idm_end << " with job_id " << dm_job_id << std::endl;

#endif
      // Get pointer to DM output data on host and on device
      dedisp_size dm_stride = 1ULL * nsamp_padded_segment * out_bytes_per_sample;
      auto *h_out = dm_job.h_data_t_dm->data();
      auto *d_out = (float *)dm_job.d_data_x_dm->data();

      // Fourier transform results back to time domain if required
      executestream->record(dm_job.postprocessingStart);

      
      for (unsigned int i = 0; i < ndm_batch_max / ndm_fft_batch; i++) {
        gpufftReal *odata =
            (gpufftReal *)d_out + i * nsamp_padded_segment * ndm_fft_batch;
        gpufftComplex *idata = (gpufftComplex *)odata;
        //C2Rtimer.Start();
        gpufftExecC2R(plan_c2r, idata, odata);
        //C2Rtimer.Stop();
        //c2rtime += C2Rtimer.Elapsed();
      }

      // FFT scaling
      kernel.scale(dm_job.ndm_current, // height
                    nsamp_padded_segment,       // width
                    nsamp_padded_segment,       // stride
                    1.0f / nsamp_fft_segment,   // scale
                    d_out,              // d_data
                    *executestream);    // stream
      
      
      executestream->record(dm_job.postprocessingEnd);

      // Copy output. If inverse FFT is not applied, the fourier coefficients are copied
      // Output is picked up by (already running) host side thread
      // and is there copied from CPU pinned to paged memory
      dm_job.gpu_lock.lock();
      dtohstream->waitEvent(dm_job.postprocessingEnd);
      dtohstream->record(dm_job.outputStart);
      dedisp_size size = 1ULL * dm_job.ndm_current * dm_stride;
      dtohstream->memcpyDtoHAsync(h_out, // dst
                                  d_out, // src
                                  size); // size
      dtohstream->record(dm_job.outputEnd);
      dm_job.cpu_lock.unlock();
    } // end for dm_job_id_inner
  } // end for dm_job_id_outer

  // Wait for final memory transfer
  // Wait for host threads to exit
  if (output_thread.joinable()) {
    output_thread.join();
  }

  if (mpi_thread.joinable()) {
    mpi_thread.join();
  }
  dtohstream->record(eEndGPU);
  mExeGPU.end(eEndGPU);
#ifdef DEDISP_BENCHMARK
  total_timer->Pause();

  gpuexec_timer->Add(eEndGPU.elapsedTime(eStartGPU));

  // Accumulate postprocessing time for all dm jobs
  for (auto &job : dm_jobs) {
    postprocessing_timer->Add(
        job.postprocessingEnd.elapsedTime(job.postprocessingStart));
    output_timer->Add(job.outputEnd.elapsedTime(job.outputStart));
  }

  // Print timings
  long double runtime_time = preprocessing_timer->Milliseconds() +
                             dedispersion_timer->Milliseconds() +
                             postprocessing_timer->Milliseconds();
  runtime_time *= 1e-3; // seconds
  std::stringstream runtime_time_string;
  runtime_time_string << std::fixed;
  runtime_time_string << runtime_time;

  std::cout << timings_str << std::endl;
  std::cout << init_time_str << init_timer->ToString() << " sec." << std::endl;
  std::cout << input_memcpy_time_str << input_timer->ToString() << " sec."
            << std::endl;
  std::cout << preprocessing_time_str << preprocessing_timer->ToString()
            << " sec." << std::endl;
  std::cout << dedispersion_time_str << dedispersion_timer->ToString()
            << " sec." << std::endl;
  std::cout << postprocessing_time_str << postprocessing_timer->ToString()
            << " sec." << std::endl;
  std::cout << output_memcpy_time_str << output_timer->ToString() << " sec."
            << std::endl;
  std::cout << runtime_time_str << runtime_time_string.str() << " sec."
            << std::endl;
  std::cout << gpuexec_time_str << gpuexec_timer->ToString() << " sec."
            << std::endl;
  std::cout << total_time_str << total_timer->ToString() << " sec."
            << std::endl;
  std::cout << std::endl;
#endif

  // Free FFT plans
  gpufftDestroy(plan_c2r);
  gpufftDestroy(plan_r2c);
  printf("R2C transforms took %.3f seconds \n", r2ctime);
  printf("C2R transforms took %.3f seconds \n", c2rtime);
}

/*    Refer to execute_gpu() above for additional comments on common constructs
 * Optional feature:
 * Input samples are divided in to nicely dimensioned
 * segments (time samples) and then processed for all DMs.
 * This allows to only copy input data to the GPU once.
 * Contrary to the alternative approach where, for large amounts of trial-DMs we
 * introduce an outer DM job to overcome GPU memory size limitations, the
 * separation in outer DM jobs requires an additional pass/passess over the
 * input data which might lead to inefficiency. However we are able to overlap
 * transfer and compute well, thus minimizing inefficiency. Also segmentation
 * allows for smaller sized (more efficient) FFTs. We are leaving this feature
 * in because the balance between the current default method (dimensioning in DM
 * outer and inner jobs) and this feature might be different depending on the
 * GPU Architecture. Note the time segmentation feature might miss very large
 * DMs when using small segments of input data.
 */
// Private interface for FDD on GPU with time segmentation of input data
void FDDGPUPlan::execute_gpu_segmented(size_type nsamps, const byte_type *in,
                                       size_type in_nbits, byte_type *out,
                                       size_type out_nbits) {
  enum {
    BITS_PER_BYTE = 8,
    BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
  };

  assert(in_nbits == 8);
  assert(out_nbits == 32);

  // Parameters
  float dt = m_dt;                      // sample time
  unsigned int nchan = m_nchans;        // number of observering frequencies
  unsigned int nsamp = nsamps;          // number of time samples
  unsigned int nfreq = (nsamp / 2 + 1); // number of spin frequencies
  unsigned int ndm = m_dm_count;        // number of DMs
  unsigned int nfft = 16384; // number of samples processed in a segment
  // nfft should be set to a mulitple of powers of 2, 3 or 5 for good cuFFT
  // performance

  // Compute the number of output samples
  unsigned int nsamp_computed = nsamp - m_max_delay;

  /* Compute the number of time segments ("chunks" hereafter):
   *  Segmentation of input samples introduces errors in the FFTed data
   *  nsamp_good denotes the good results, the other results are unused
   *  thus creating an inefficiency.
   *  The inefficiency might be acceptable depending on the cost of a.o.:
   *  - copy of input data
   *  - input data size (nfft)
   *  - efficiency of the FFT (nfft)
   *  - GPU memory size (nchan_ and ndm_ buffers)
   *  - Balance between nfft and nsamp_dm
   *  Here nfft is tuned to a specified minimal efficiency (min_efficiency)
   *  nchunk is based on the number of good samples (nsamp_good)
   */
  unsigned int nsamp_dm = std::ceil(m_max_delay);
  float min_efficiency = 0.8;
  while ((nfft * (1.0 - min_efficiency)) < nsamp_dm) {
    nfft *= 2;
  };
  unsigned int nsamp_good = nfft - nsamp_dm;
  unsigned int nchunk = std::ceil((float)nsamp / nsamp_good);

  // For every channel, a buffer of nsamp_padded scalar elements long is used,
  // resulting in a two-dimensional buffers of size buffer[nchan][nsamp_padded]
  // Every row of is divided into chunks of nfreq_chunk_padded complex elements,
  // thus the implicit dimensions are buffer[nchan][nchunk][nfreq_chunk_padded],
  // of which only nfreq_chunk elements in the innermost dimension are used.
  unsigned int nfreq_chunk = std::ceil(nfft / 2) + 1;
  unsigned int nfreq_chunk_padded = round_up(nfreq_chunk + 1, 1024);
  unsigned int nsamp_padded = nchunk * (nfreq_chunk_padded * 2);

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

  // Maximum number of DMs computed in one gulp
  unsigned int ndm_batch_max = 32;
  unsigned int ndm_buffers = 8;
  ndm_buffers = std::min(ndm_buffers,
                         (unsigned int)((ndm + ndm_batch_max) / ndm_batch_max));

  // Maximum number of channels processed in one gulp
  unsigned int nchan_batch_max = 32;
  unsigned int nchan_buffers = 2;

  // Verbose iteration reporting
#ifdef DEDISP_DEBUG
  bool enable_verbose_iteration_reporting = true;
#endif

  // Compute derived counts
  dedisp_size out_bytes_per_sample =
      out_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);
  dedisp_size chans_per_word = sizeof(dedisp_word) * BITS_PER_BYTE / in_nbits;

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
#ifdef DEDISP_BENCHMARK
  std::unique_ptr<Stopwatch> init_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> input_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> preprocessing_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> dedispersion_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> postprocessing_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> output_timer(Stopwatch::create());
  std::unique_ptr<Stopwatch> total_timer(Stopwatch::create());
  total_timer->Start();
  init_timer->Start();
#endif

  /* Allocate memory
   *  nchan_buffers and ndm_buffers might be made automatic tuning parameters.
   *  When used in production one should add error checking on overallocating
   * memory.
   */
#ifdef DEDISP_DEBUG
  std::cout << memory_alloc_str << std::endl;
#endif
  mAllocMem.start();
  cu::HostMemory h_data_t_dm(ndm * nsamp_padded * sizeof(float));
  cu::DeviceMemory d_data_t_nu(nchan_batch_max * nsamp_padded * sizeof(float));
  cu::DeviceMemory d_data_f_nu(nchan_batch_max * nsamp_padded * sizeof(float));
  std::vector<cu::HostMemory> h_data_t_nu_(nchan_buffers);
  std::vector<cu::DeviceMemory> d_data_t_nu_(nchan_buffers);
  std::vector<cu::DeviceMemory> d_data_f_dm_(ndm_buffers);
  std::vector<cu::DeviceMemory> d_data_t_dm_(ndm_buffers);
  for (unsigned int i = 0; i < nchan_buffers; i++) {
    h_data_t_nu_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
    d_data_t_nu_[i].resize(nsamp * nchan_words_gulp * sizeof(dedisp_word));
  }
  for (unsigned int i = 0; i < ndm_buffers; i++) {
    d_data_f_dm_[i].resize(ndm_batch_max * nsamp_padded * sizeof(float));
    d_data_t_dm_[i].resize(ndm_batch_max * nsamp_padded * sizeof(float));
  }
  mAllocMem.end();

  // Prepare cuFFT plans
#ifdef DEDISP_DEBUG
  std::cout << fft_plan_str << std::endl;
#endif
  mPrepFFT.start();
  gpufftHandle plan_r2c, plan_c2r;
  int n[] = {(int)nfft};
  std::thread thread_r2c = std::thread([&]() {
    int inembed[] = {(int)nsamp_good};
    int onembed[] = {(int)nfreq_chunk_padded};
    gpufftResult result =
        gpufftPlanMany(&plan_r2c,              // plan
                      1, n,                   // rank, n
                      inembed, 1, inembed[0], // inembed, istride, idist
                      onembed, 1, onembed[0], // onembed, ostride, odist
                      GPUFFT_R2C,              // type
                      nchunk);                // batch
    if (result != GPUFFT_SUCCESS) {
      throw std::runtime_error("Error creating real to complex FFT plan.");
    }
    gpufftSetStream(plan_r2c, *executestream);
  });
  std::thread thread_c2r = std::thread([&]() {
    int inembed[] = {(int)nfreq_chunk_padded};
    int onembed[] = {(int)nfreq_chunk_padded * 2};
    gpufftResult result =
        gpufftPlanMany(&plan_c2r,              // plan
                      1, n,                   // rank, n
                      inembed, 1, inembed[0], // inembed, istride, idist
                      onembed, 1, onembed[0], // onembed, ostride, odist
                      GPUFFT_C2R,              // type
                      nchunk);                // batch
    if (result != GPUFFT_SUCCESS) {
      throw std::runtime_error("Error creating complex to real FFT plan.");
    }
    gpufftSetStream(plan_c2r, *executestream);
  });

  // Compute chunks
  std::vector<Chunk> chunks(nchunk);
  unsigned int nfreq_computed;
  compute_chunks(nsamp, nsamp_good, nfft, nfreq_chunk_padded, nfreq_computed,
                 chunks);

  // Wait for cuFFT plans to be created
  if (thread_r2c.joinable()) {
    thread_r2c.join();
  }
  if (thread_c2r.joinable()) {
    thread_c2r.join();
  }
  mPrepFFT.end();

  // Generate spin frequency table
  mPrepSpinf.start();
  if (h_spin_frequencies.size() != nsamp_padded) {
    // Generate spin frequencies on the host
    h_spin_frequencies.resize(nsamp_padded);
    generate_spin_frequency_table_chunks(
        chunks, h_spin_frequencies, nfreq_chunk, nfreq_chunk_padded, nfft, dt);

    // Copy segmented spin frequencies to the GPU
    d_spin_frequencies.resize(h_spin_frequencies.size() * sizeof(float));
    htodstream->memcpyHtoDAsync(d_spin_frequencies, h_spin_frequencies.data(),
                                d_spin_frequencies.size());
  }
  mPrepSpinf.end();

  // Initialize FDDKernel
  FDDKernel kernel;
  mDelayTable.start();
  kernel.copy_delay_table(d_delay_table, m_nchans * sizeof(dedisp_float), 0,
                          *htodstream);
  mDelayTable.end();
#ifdef DEDISP_BENCHMARK
  init_timer->Pause();
#endif

  struct ChannelData {
    unsigned int ichan_start;
    unsigned int ichan_end;
    unsigned int nchan_current;
    void *h_in_ptr;
    void *d_in_ptr;
    cu::Event inputStart, inputEnd;
    cu::Event preprocessingStart, preprocessingEnd;
    cu::Event outputStart, outputEnd;
  };

  unsigned int nchan_jobs = (nchan + nchan_batch_max) / nchan_batch_max;
  std::vector<ChannelData> channel_jobs(nchan_jobs);

  for (unsigned job_id = 0; job_id < nchan_jobs; job_id++) {
    ChannelData &job = channel_jobs[job_id];
    job.ichan_start = job_id == 0 ? 0 : channel_jobs[job_id - 1].ichan_end;
    job.nchan_current = std::min(nchan_batch_max, nchan - job.ichan_start);
    job.ichan_end = job.ichan_start + job.nchan_current;
    job.h_in_ptr = h_data_t_nu_[job_id % nchan_buffers];
    job.d_in_ptr = d_data_t_nu_[job_id % nchan_buffers];
    if (job.nchan_current == 0) {
      channel_jobs.pop_back();
    }
  }

  struct DMData {
    unsigned int idm_start;
    unsigned int idm_end;
    unsigned int ndm_current;
    float *h_in_ptr;
    dedisp_float2 *d_data_f_dm_ptr;
    dedisp_float2 *d_data_t_dm_ptr;
    cu::Event inputStart, inputEnd;
    cu::Event dedispersionStart, dedispersionEnd;
    cu::Event postprocessingStart, postprocessingEnd;
    cu::Event outputStart, outputEnd;
  };

  unsigned int ndm_jobs = (ndm + ndm_batch_max) / ndm_batch_max;
  std::vector<DMData> dm_jobs(ndm_jobs);

  for (unsigned job_id = 0; job_id < ndm_jobs; job_id++) {
    DMData &job = dm_jobs[job_id];
    job.idm_start = job_id == 0 ? 0 : dm_jobs[job_id - 1].idm_end;
    job.ndm_current = std::min(ndm_batch_max, ndm - job.idm_start);
    job.idm_end = job.idm_start + job.ndm_current;
    job.d_data_f_dm_ptr = d_data_f_dm_[job_id % ndm_buffers];
    job.d_data_t_dm_ptr = d_data_t_dm_[job_id % ndm_buffers];
    if (job.ndm_current == 0) {
      dm_jobs.pop_back();
    }
  }
#ifdef DEDISP_DEBUG
  std::cout << fdd_dedispersion_str << std::endl;
#endif
  htodstream->record(eStartGPU);
  mExeGPU.start();

  // Process all dm batches
  for (unsigned dm_job_id_outer = 0; dm_job_id_outer < dm_jobs.size();
       dm_job_id_outer += ndm_buffers) {
    // Process all channel batches
    for (unsigned channel_job_id = 0; channel_job_id < channel_jobs.size();
         channel_job_id++) {
      auto &channel_job = channel_jobs[channel_job_id];
#ifdef DEDISP_DEBUG
      // Info
      if (enable_verbose_iteration_reporting) {
        std::cout << "Processing channel " << channel_job.ichan_start << " to "
                  << channel_job.ichan_end << std::endl;
      }
#endif
      // Channel input size
      dedisp_size dst_stride = nchan_words_gulp * sizeof(dedisp_word);
      dedisp_size src_stride = nchan_words * sizeof(dedisp_word);

      // Copy the input data for the first job
      if (channel_job_id == 0) {
        dedisp_size gulp_chan_byte_idx =
            (channel_job.ichan_start / chans_per_word) * sizeof(dedisp_word);
        memcpy2D(channel_job.h_in_ptr,    // dst
                 dst_stride,              // dst width
                 in + gulp_chan_byte_idx, // src
                 src_stride,              // src width
                 dst_stride,              // width bytes
                 nsamp);                  // height
        htodstream->record(channel_job.inputStart);
        htodstream->memcpyHtoDAsync(channel_job.d_in_ptr, // dst
                                    channel_job.h_in_ptr, // src
                                    nsamp * dst_stride);  // size
        htodstream->record(channel_job.inputEnd);
      }
      executestream->waitEvent(channel_job.inputEnd);

      // Transpose and upack the data
      executestream->record(channel_job.preprocessingStart);
      transpose_unpack((float *)channel_job.d_in_ptr, // d_in
                       nchan_words_gulp,                    // input width
                       nsamp,                               // input height
                       nchan_words_gulp,                    // in_stride
                       nsamp_padded,                        // out_stride
                       d_data_t_nu,                         // d_out
                       in_nbits, 32,    // in_nbits, out_nbits
                       1.0 / nchan,     // scale
                       *executestream); // stream

      // Apply zero padding
      auto dst_ptr = ((float *)d_data_t_nu.data()) + nsamp;
      unsigned int nsamp_padding = nsamp_padded - nsamp;
      gpuMemset2DAsync(dst_ptr,                       // devPtr
                                       nsamp_padded * sizeof(float),  // pitch
                                       0,                             // value
                                       nsamp_padding * sizeof(float), // width
                                       nchan_batch_max,               // height
                                       *executestream);

      // FFT data (real to complex) along time axis
      for (unsigned int ichan = 0; ichan < channel_job.nchan_current; ichan++) {
        auto *idata =
            (gpufftReal *)d_data_t_nu.data() + (1ULL * ichan * nsamp_padded);
        auto *odata = (gpufftComplex *)d_data_f_nu.data() +
                      (1ULL * ichan * nsamp_padded / 2);
        gpufftExecR2C(plan_r2c, idata, odata);
      }
      executestream->record(channel_job.preprocessingEnd);

      // Initialize output to zero
      if (channel_job_id == 0) {
        // Wait for all previous output copies to finish
        dtohstream->synchronize();

        for (cu::DeviceMemory &d_data_out : d_data_f_dm_) {
          // Use executestream to make sure dedispersion
          // starts only after initializing the output buffer
          d_data_out.zero(*executestream);
        }
      }

      // Process DM batches
      for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers;
           dm_job_id_inner++) {
        unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
        if (dm_job_id >= dm_jobs.size()) {
          break;
        }
        auto &dm_job = dm_jobs[dm_job_id];
#ifdef DEDISP_DEBUG
        // Info
        if (enable_verbose_iteration_reporting) {
          std::cout << "Processing DM " << dm_job.idm_start << " to "
                    << dm_job.idm_end << std::endl;
        }
#endif
        // Wait for temporary output from previous job to be copied
        if (channel_job_id > (nchan_buffers - 1)) {
          auto &job_previous = channel_jobs[channel_job_id - nchan_buffers];
          job_previous.outputEnd.synchronize();
        }

        // Dedispersion in frequency domain
        executestream->record(dm_job.dedispersionStart);
        kernel.launch(dm_job.ndm_current,        // ndm
                      nfreq,                     // nfreq
                      channel_job.nchan_current, // nchan
                      dt,                        // dt
                      d_spin_frequencies,        // d_spin_frequencies
                      d_dm_list,                 // d_dm_list
                      d_data_f_nu,               // d_in
                      dm_job.d_data_f_dm_ptr,    // d_out
                      nsamp_padded / 2,          // in stride
                      nsamp_padded / 2,          // out stride
                      dm_job.idm_start,          // idm_start
                      dm_job.idm_end,            // idm_end
                      channel_job.ichan_start,   // ichan_start
                      *executestream);           // stream
        executestream->record(dm_job.dedispersionEnd);
      } // end for dm_job_id_inner

      // Copy the input data for the next job (if any)
      unsigned channel_job_id_next = channel_job_id + 1;
      if (channel_job_id_next < channel_jobs.size()) {
        auto &channel_job_next = channel_jobs[channel_job_id_next];
        dedisp_size gulp_chan_byte_idx =
            (channel_job_next.ichan_start / chans_per_word) *
            sizeof(dedisp_word);
        memcpy2D(channel_job_next.h_in_ptr, // dst
                 dst_stride,                // dst width
                 in + gulp_chan_byte_idx,   // src
                 src_stride,                // src width
                 dst_stride,                // width bytes
                 nsamp);                    // height
        htodstream->record(channel_job_next.inputStart);
        htodstream->memcpyHtoDAsync(channel_job_next.d_in_ptr, // dst
                                    channel_job_next.h_in_ptr, // src
                                    nsamp * dst_stride);       // size
        htodstream->record(channel_job_next.inputEnd);
      }

      // Wait for current batch to finish
      executestream->synchronize();

      // Add input and preprocessing time for the current channel job
#ifdef DEDISP_BENCHMARK
      input_timer->Add(
          channel_job.inputEnd.elapsedTime(channel_job.inputStart));
      preprocessing_timer->Add(channel_job.preprocessingEnd.elapsedTime(
          channel_job.preprocessingStart));
#endif

      // Add dedispersion time for current dm jobs
#ifdef DEDISP_BENCHMARK
      for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers;
           dm_job_id_inner++) {
        unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
        if (dm_job_id >= dm_jobs.size()) {
          break;
        }
        auto &dm_job = dm_jobs[dm_job_id];
        dedispersion_timer->Add(
            dm_job.dedispersionEnd.elapsedTime(dm_job.dedispersionStart));
      }
#endif
    } // end for ichan_start

    // Output DM batches
    for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers;
         dm_job_id_inner++) {
      unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
      if (dm_job_id >= dm_jobs.size()) {
        break;
      }
      auto &dm_job = dm_jobs[dm_job_id];

      // Get pointer to DM output data on host and on device
      dedisp_size dm_stride = nsamp_padded * out_bytes_per_sample;
      dedisp_size dm_offset = dm_job.idm_start * dm_stride;
      auto *h_data_t_dm_ptr =
          (void *)(((size_t)h_data_t_dm.data()) + dm_offset);
      auto *d_data_f_dm_ptr = (float *)dm_job.d_data_f_dm_ptr;
      auto *d_data_t_dm_ptr = (float *)dm_job.d_data_t_dm_ptr;

      // Fourier transform results back to time domain
      executestream->record(dm_job.postprocessingStart);
      for (unsigned int idm = 0; idm < dm_job.ndm_current; idm++) {
        auto *idata =
            (gpufftComplex *)d_data_f_dm_ptr + (1ULL * idm * nsamp_padded / 2);
        auto *odata =
            (gpufftReal *)d_data_t_dm_ptr + (1ULL * idm * nsamp_padded);
        gpufftExecC2R(plan_c2r, idata, odata);
      }

      // FFT scaling
      kernel.scale(dm_job.ndm_current, // height
                   nsamp_padded,       // width
                   nsamp_padded,       // stride
                   1.0f / nfft,        // scale
                   d_data_t_dm_ptr,    // d_data
                   *executestream);    // stream
      executestream->record(dm_job.postprocessingEnd);

      // Copy output
      dtohstream->waitEvent(dm_job.postprocessingEnd);
      dtohstream->record(dm_job.outputStart);
      dtohstream->memcpyDtoHAsync(h_data_t_dm_ptr,                 // dst
                                  d_data_t_dm_ptr,                 // src
                                  dm_job.ndm_current * dm_stride); // size
      dtohstream->record(dm_job.outputEnd);
    } // end for dm_job_id_inner
  } // end for dm_job_id_outer

  // Wait for final memory transfer
  dtohstream->record(eEndGPU);
  mExeGPU.end(eEndGPU);
  dtohstream->synchronize();

  // Copy output
#ifdef DEDISP_DEBUG
  std::cout << copy_output_str << std::endl;
#endif
  mCopyMem.start();
#ifdef DEDISP_BENCHMARK
  output_timer->Start();
#endif
  copy_chunk_output((float *)h_data_t_dm.data(), (float *)out, ndm, nsamp,
                    nsamp_computed, nsamp_padded, nsamp_good, chunks);
#ifdef DEDISP_BENCHMARK
  output_timer->Pause();
#endif
  mCopyMem.end();
#ifdef DEDISP_BENCHMARK
  total_timer->Pause();

  // Accumulate dedispersion and postprocessing time for all dm jobs
  for (auto &job : dm_jobs) {
    postprocessing_timer->Add(
        job.postprocessingEnd.elapsedTime(job.postprocessingStart));
  }

  // Print timings
  long double runtime_time = preprocessing_timer->Milliseconds() +
                             dedispersion_timer->Milliseconds() +
                             postprocessing_timer->Milliseconds();
  runtime_time *= 1e-3; // seconds
  std::stringstream runtime_time_string;
  runtime_time_string << std::fixed;
  runtime_time_string << runtime_time;

  std::cout << timings_str << std::endl;
  std::cout << init_time_str << init_timer->ToString() << " sec." << std::endl;
  std::cout << input_memcpy_time_str << input_timer->ToString() << " sec."
            << std::endl;
  std::cout << preprocessing_time_str << preprocessing_timer->ToString()
            << " sec." << std::endl;
  std::cout << dedispersion_time_str << dedispersion_timer->ToString()
            << " sec." << std::endl;
  std::cout << postprocessing_time_str << postprocessing_timer->ToString()
            << " sec." << std::endl;
  std::cout << output_memcpy_time_str << output_timer->ToString() << " sec."
            << std::endl;
  std::cout << runtime_time_str << runtime_time_string.str() << " sec."
            << std::endl;
  std::cout << total_time_str << total_timer->ToString() << " sec."
            << std::endl;
  std::cout << std::endl;
#endif

  // Free FFT plans
  gpufftDestroy(plan_c2r);
  gpufftDestroy(plan_r2c);
}

// Private helper function
void FDDGPUPlan::generate_spin_frequency_table(dedisp_size nfreq,
                                               dedisp_size nsamp_fft,
                                               dedisp_float dt) {
  h_spin_frequencies.resize(nfreq);

#pragma omp parallel for
  for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++) {
    // Skipping the scaling of 1/dt in the spin frequencies which saves a multiplication
    // in the GPU kernel
    h_spin_frequencies[ifreq] = ifreq * (1.0 / (nsamp_fft));
  }

  d_spin_frequencies.resize(nfreq * sizeof(dedisp_float));

  htodstream->memcpyHtoDAsync(d_spin_frequencies, h_spin_frequencies.data(),
                              d_spin_frequencies.size());
}

} // end namespace dedisp