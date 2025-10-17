// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include "FDDGPUPlan.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include <assert.h>
#include <cufft.h>
#include <omp.h>

#include "common/dedisp_strings.h"
#include "dedisperse/FDDKernel.hpp"
#include "unpack/unpack.h"
//#define DEDISP_BENCHMARK
#ifdef DEDISP_BENCHMARK
#include "external/Stopwatch.h"
#endif

#include "chunk.h"
#include "common/helper.h"
#include "helper.h"
#include "fdd_gpu.h"
#include "mpi.h"
#include "nccl_macro.hpp"

#ifdef USECUFILE
#include <cufile.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <nvToolsExt.h>

size_t nsamp_fft = 0;
size_t nsamps_computed = 0;
size_t nsamp_padded = 0;
int numGPUsLocal = 0;
int total_cores = 0;

namespace dedisp {

void FDDGPUPlan::gpuDirectWrite(std::string path_string, void* d_buf, size_t size) {
  #ifdef USECUFILE
  const off_t file_offset = 0;                      
  const char* path = path_string.c_str();

  // 1) Init cuFile driver
  CK_CUFILE(cuFileDriverOpen());

  // 2) Open file with O_DIRECT
  int fd = open(path, O_CREAT | O_WRONLY | O_DIRECT, 0664);
  if (fd < 0) {
      perror("open(O_DIRECT)");
      return;
  }

  // 3) Register file handle with cuFile
  CUfileDescr_t desc;
  memset(&desc, 0, sizeof(desc));
  desc.handle.fd = fd;
  desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

  CUfileHandle_t cf_handle;
  CK_CUFILE(cuFileHandleRegister(&cf_handle, &desc));

  // 5) (Recommended) Register the GPU buffer with cuFile for best performance
  CK_CUFILE(cuFileBufRegister(d_buf, size, 0));

  size_t written = cuFileWrite(cf_handle,
                                d_buf,
                                size,     // length (aligned)
                                file_offset,  // file offset (aligned)
                                0);           // device buffer offset

  if (written < 0) {
      fprintf(stderr, "cuFileWrite failed: %zd\n", written);
      // Clean up before exit
      cuFileBufDeregister(d_buf);
      cuFileHandleDeregister(cf_handle);
      close(fd);
      cuFileDriverClose();
      return;
  }

  // 7) Cleanup
  CK_CUFILE(cuFileBufDeregister(d_buf));
  cuFileHandleDeregister(cf_handle);
  close(fd);
  CK_CUFILE(cuFileDriverClose());

  printf("Wrote %zd bytes (GPU->file) to %s\n", written, path);
  #endif
} 

// Constructor
FDDGPUPlan::FDDGPUPlan(size_type nchans, 
                       float_type dt, 
                       float_type f0,
                       float_type df, 
                       int device_idx, 
                       int global_gpu_id_, 
                       int start_chan_,
                       int end_chan_, 
                       int nchan_gpu_, 
                       int start_chan_node_, 
                       ncclComm_t comm_)
    : GPUPlan(nchans, dt, f0, df, device_idx) {
      global_gpu_id = global_gpu_id_;
      local_gpu_id = device_idx;
      start_chan = start_chan_;
      end_chan = end_chan_;
      nchan_gpu = nchan_gpu_;
      start_chan_node = start_chan_node_;
      comm = comm_;
    }

// Destructor
FDDGPUPlan::~FDDGPUPlan() {}

// Assumes same height of the source and destination memory layout in 2D
void FDDGPUPlan::memcpy2D_internal(
    void *dstPtr, size_t dstWidth,
    const void *srcPtr, size_t srcWidth,
    size_t widthBytes, size_t height)
{
    cu::Marker mCopyMem("memcpy2D_internal", cu::Marker::Color::blue);
    mCopyMem.start();
    typedef char SrcType[height][srcWidth];
    typedef char DstType[height][dstWidth];

    auto src = (SrcType *) srcPtr;
    auto dst = (DstType *) dstPtr;

    int memcpy2d_cores = total_cores / numGPUsLocal - 2;
    #pragma omp parallel
    {
      int thread_id = omp_get_thread_num();
      pin_thread_to_core(2 * numGPUsLocal + local_gpu_id * memcpy2d_cores + thread_id);

      #pragma omp for
      for (size_t y = 0; y < height; y++)
      {
          /* for (size_t x = 0; x < widthBytes; x++)
          {
              (*dst)[y][x] = (*src)[y][x];
          } */
         std::memcpy((*dst)[y], (*src)[y], widthBytes);
      }
    }
    mCopyMem.end();
    
}

// Public interface for FDD on GPU
void FDDGPUPlan::execute(size_type nsamps, const byte_type *in,
                         size_type in_nbits, byte_type *out,
                         size_type out_nbits, unsigned flags) {
  char *use_segmented_str = getenv("USE_SEGMENTED");
  bool use_segmented = !use_segmented_str ? false : atoi(use_segmented_str);
  if (use_segmented) {
#ifdef DEDISP_DEBUG
    std::cout << ">> Running segmented GPU implementation" << std::endl;
#endif
    execute_gpu_segmented(nsamps, in, in_nbits, out, out_nbits);
  } else { // Default
#ifdef DEDISP_DEBUG
    std::cout << ">> Running GPU implementation" << std::endl;
#endif
    execute_gpu(nsamps, in, in_nbits, out, out_nbits);
  }
}

void FDDGPUPlan::determineBufferSizes(size_t nsamps) {
  
  nchan_fft_batch = 64;
  nchan = nchan_gpu;        // number of channels for this GPU
  nsamp = nsamps;          // number of time samples
  nfreq = (nsamp_fft / 2 + 1); // number of spin frequencies
  ndm = m_dm_count;        // number of DMs

  // Maximum number of DMs computed in one gulp
  // Parameters might be tuned for efficiency depending on system architecture
  ndm_batch_max = std::min(ndm / 4, (unsigned int)64);
  ndm_fft_batch = 32;
  ndm_fft_batch = std::min(ndm_batch_max, ndm_fft_batch);
  // The number of buffers for DM results is configured below based on the
  // amount of available GPU memory.
  ndm_buffers = 1;

  // Maximum number of channels processed in one gulp
  // Parameters might be tuned for efficiency depending on system architecture
  nchan_batch_max = std::min(nchan / 4, (unsigned int)64);
  
  nchan_buffers = 2;

  chans_per_word = 1;

  /* std::cout << "nsamp        = " << nsamp << std::endl;
  std::cout << "nsamp_fft    = " << nsamp_fft << std::endl;
  std::cout << "nsamp_padded = " << nsamp_padded << std::endl;
  std::cout << "m_max_delay = " << m_max_delay << std::endl;
  std::cout << "nsamps_computed = " << nsamps_computed << std::endl; */

  // The number of channel words in the input
  // For input that contains all channels per node, distribution between GPUs
  // requires stride of nchans (total)
  nchan_words = m_nchans / chans_per_word;

  // The number of channel words proccessed in one gulp
  nchan_words_gulp = nchan_batch_max / chans_per_word;

  size_t d_memory_total = m_device->get_total_memory();
  size_t d_memory_free = m_device->get_free_memory();

  sizeof_data_t_nu =
      1ULL * nsamp * nchan_words_gulp * sizeof(dedisp_word);
  sizeof_data_x_nu =
      1ULL * nchan_batch_max * nsamp_padded * sizeof(float);
  sizeof_data_x_dm = 1ULL * ndm_batch_max * nsamp_padded * sizeof(float);
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

  /* std::cout << debug_str << std::endl;
  std::cout << "ndm_buffers     = " << ndm_buffers << " x " << ndm_batch_max
            << " DMs" << std::endl;
  std::cout << "nchan_buffers   = " << nchan_buffers << " x " << nchan_batch_max
            << " channels" << std::endl;
  std::cout << "Device memory total    = " << d_memory_total / std::pow(1024, 3)
            << " Gb" << std::endl;
  std::cout << "Device memory free     = " << d_memory_free / std::pow(1024, 3)
            << " Gb" << std::endl;
  std::cout << "Device Memory required = "
            << d_memory_required / std::pow(1024, 3) << " Gb" << std::endl;
  std::cout << "Host memory total    = "
            << get_total_memory() / std::pow(1024, 1) << " Gb" << std::endl;
  std::cout << "Host memory free     = "
            << get_free_memory() / std::pow(1024, 1) << " Gb" << std::endl; */
}

void FDDGPUPlan::allocateMem(size_t nsamps) {
  // This function just allocates the host/device memory required for the execute step
  // The memory to be allocated is calculated based on a few parameters
  cudaSetDevice(local_gpu_id);

  
  int n[] = {(int)nsamp_fft};
  int rnembed[] = {(int)nsamp_padded};     // width in real elements
  int cnembed[] = {(int)nsamp_padded / 2}; // width in complex elements
  
  // Threads for cufft plan r2c
  //std::jthread thread_r2c = std::jthread([&]() {
  //  cudaSetDevice(local_gpu_id);
    cu::Marker mPrepFFT("cufft Plan r2c", cu::Marker::yellow);
    mPrepFFT.start();

    cufftResult result =
        cufftPlanMany(&plan_r2c,              // plan
                      1, n,                   // rank, n
                      rnembed, 1, rnembed[0], // inembed, istride, idist
                      cnembed, 1, cnembed[0], // onembed, ostride, odist
                      CUFFT_R2C,              // type
                      nchan_fft_batch);       // batch
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Error creating real to complex FFT plan.");
    }
    cufftSetStream(plan_r2c, *executestream);
    mPrepFFT.end();
  //});

  // Threads for cufft plan c2r
  //std::jthread thread_c2r = std::jthread([&]() {
  //  cudaSetDevice(local_gpu_id);
    cu::Marker mPrepFFT1("cufft Plan c2r", cu::Marker::yellow);
    mPrepFFT1.start();

      result =
        cufftPlanMany(&plan_c2r,              // plan
                      1, n,                   // rank, n
                      cnembed, 1, cnembed[0], // inembed, istride, idist
                      rnembed, 1, rnembed[0], // onembed, ostride, odist
                      CUFFT_C2R,              // type
                      ndm_fft_batch);         // batch
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Error creating complex to real FFT plan.");
    }
    cufftSetStream(plan_c2r, *executestream);
    mPrepFFT1.end();
  //});

  cu::Marker mAllocMem("Allocate host and device memory", cu::Marker::black);
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

  // Allocate host output buffers if not using direct storage for output
  #ifndef USECUFILE
  // Allocate host output buffers only once per node
  if (local_gpu_id == 0) {
    h_data_t_dm_.resize(ndm_buffers);
  }
  #endif
  //h_data_t_nu_.resize(nchan_buffers);
  
  d_data_t_nu_.resize(nchan_buffers);
  d_data_x_dm_.resize(ndm_buffers);

  //cu::DeviceMemory d_data_x_nu(sizeof_data_x_nu);
  d_data_x_nu.resize(sizeof_data_x_nu);

  for (unsigned int i = 0; i < nchan_buffers; i++) {
    //h_data_t_nu_[i].resize(sizeof_data_t_nu);
    d_data_t_nu_[i].resize(sizeof_data_t_nu);
  }

  for (unsigned int i = 0; i < ndm_buffers; i++) {
    #ifdef USECUFILE
    d_data_x_dm_[i].resize(round_up_4k(sizeof_data_x_dm));
    #else
    if (local_gpu_id == 0)
      h_data_t_dm_[i].resize(sizeof_data_x_dm);
    d_data_x_dm_[i].resize(sizeof_data_x_dm);
    #endif
  }
  mAllocMem.end();

}

// Private interface for FDD on GPU
void FDDGPUPlan::execute_gpu(size_type nsamps, const byte_type *in,
                             size_type in_nbits, byte_type *out,
                             size_type out_nbits) {
  cudaSetDevice(local_gpu_id);
  cudaFree(0); 

  /* int n[] = {(int)nsamp_fft};
  int rnembed[] = {(int)nsamp_padded};     // width in real elements
  int cnembed[] = {(int)nsamp_padded / 2}; // width in complex elements
  
  // Threads for cufft plan r2c
  //std::jthread thread_r2c = std::jthread([&]() {
  //  cudaSetDevice(local_gpu_id);
    cu::Marker mPrepFFT("cufft Plan r2c", cu::Marker::yellow);
    mPrepFFT.start();

    cufftResult result =
        cufftPlanMany(&plan_r2c,              // plan
                      1, n,                   // rank, n
                      rnembed, 1, rnembed[0], // inembed, istride, idist
                      cnembed, 1, cnembed[0], // onembed, ostride, odist
                      CUFFT_R2C,              // type
                      nchan_fft_batch);       // batch
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Error creating real to complex FFT plan.");
    }
    cufftSetStream(plan_r2c, *executestream);
    mPrepFFT.end();
  //});

  // Threads for cufft plan c2r
  //std::jthread thread_c2r = std::jthread([&]() {
  //  cudaSetDevice(local_gpu_id);
    cu::Marker mPrepFFT1("cufft Plan c2r", cu::Marker::yellow);
    mPrepFFT1.start();

      result =
        cufftPlanMany(&plan_c2r,              // plan
                      1, n,                   // rank, n
                      cnembed, 1, cnembed[0], // inembed, istride, idist
                      rnembed, 1, rnembed[0], // onembed, ostride, odist
                      CUFFT_C2R,              // type
                      ndm_fft_batch);         // batch
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Error creating complex to real FFT plan.");
    }
    cufftSetStream(plan_c2r, *executestream);
    mPrepFFT1.end(); */

  enum {
    BITS_PER_BYTE = 8,
    BYTES_PER_WORD = sizeof(dedisp_word) / sizeof(dedisp_byte)
  };

  int loc_gpu_id = local_gpu_id;

  float* float_in = (float*) in;

  aa_gpu_timer C2Rtimer;
  double c2rtime = 0;

  aa_gpu_timer R2Ctimer;
  double r2ctime = 0;

  // Original code. Commented out to allow working directly with floats
  //assert(in_nbits == 8);
  assert(out_nbits == 32);

  // Parameters
  float dt = m_dt;                      // sample time
  
#ifdef DEDISP_DEBUG
  std::cout << debug_str << std::endl;
  std::cout << "nsamp_fft    = " << nsamp_fft << std::endl;
  std::cout << "nsamp_padded = " << nsamp_padded << std::endl;
#endif

  // Verbose iteration reporting
#ifdef DEDISP_DEBUG
  bool enable_verbose_iteration_reporting = false;
#endif

  // Compute derived counts
  dedisp_size out_bytes_per_sample =
      out_nbits / (sizeof(dedisp_byte) * BITS_PER_BYTE);

  // Events, markers, timers
  cu::Event eStartGPU, eEndGPU;
  cu::Marker mCopyMem("Copy CUDA mem to CPU mem", cu::Marker::black);
  cu::Marker mPrepSpinf("spin Frequency generation", cu::Marker::blue);
  cu::Marker mDelayTable("Delay table copy", cu::Marker::black);
  cu::Marker mExeGPU("Dedisp fdd execution on GPU", cu::Marker::green);
  cu::Marker m1("Finishing #ndm_buffer dedispersion kernels, moving to next channel batch", cu::Marker::green);
  cu::Marker m2("DM job prev. outputEnd", cu::Marker::green);
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

  if ((nsamp_fft - (int) nsamp_fft) != 0 || (nsamp_padded - (int) nsamp_padded) != 0) {
    std::cerr << "Too large sizes for nsamp_fft or nsamp_padded. Check FDDGPUPlan.cpp" << std::endl;
    exit(1);
  }

  // Prepare cuFFT plans
#ifdef DEDISP_DEBUG
  std::cout << fft_plan_str << std::endl;
#endif

  int genspinfreq_cores = total_cores / numGPUsLocal - 1;
  if (genspinfreq_cores < 1) {
    std::cerr << "Not enough CPU cores for generating spin frequency table!" << std::endl;
    exit(-1);
  }
  // Set omp num threads, ensuring proper division of the available cores into gpus
  omp_set_num_threads(genspinfreq_cores);

  // Generate spin frequency table
  mPrepSpinf.start();
  if (h_spin_frequencies.size() != nfreq) {
    generate_spin_frequency_table(nfreq, nsamp_fft, dt);
  }
  mPrepSpinf.end();

  // Set omp num threads, ensuring proper division of the available cores into gpus
  // This one is done for the memcpy2D_internal function
  int memcpy2d_cores = total_cores / numGPUsLocal - 2;
  if (memcpy2d_cores < 1) {
    std::cerr << "Not enough CPU cores for memcpy2D_internal!" << std::endl;
    exit(-1);
  }
  omp_set_num_threads(genspinfreq_cores);

  // Allocate memory
#ifdef DEDISP_DEBUG
  std::cout << memory_alloc_str << std::endl;
#endif


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
  // The delay table is for all the channels treated globally
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
    job.ichan_start = job_id == 0 ? start_chan : channel_jobs[job_id - 1].ichan_end;
    job.nchan_current = std::min(nchan_batch_max, end_chan - job.ichan_start);
    job.ichan_end = job.ichan_start + job.nchan_current;
    //job.h_in_ptr = h_data_t_nu_[job_id % nchan_buffers];
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

    #ifndef USECUFILE
    if (local_gpu_id == 0)
      job.h_data_t_dm = &h_data_t_dm_[job_id % ndm_buffers];
    #endif
    
    job.d_data_x_dm = &d_data_x_dm_[job_id % ndm_buffers];
    if (job.ndm_current == 0) {
      dm_jobs.pop_back();
    }
    job.cpu_lock.lock();
    if (job_id > ndm_buffers) {
      job.gpu_lock.lock();
    }
  }

  std::cout << "ndm_jobs = " << ndm_jobs << ", dm_jobs.size() = " << dm_jobs.size() << std::endl;

  #ifndef USECUFILE
  std::thread output_thread;
  // Take care of the output only once per node
  // Launch thread to copy output data from device to host for each dm_job
  output_thread = std::thread([&]() {
    pin_thread_to_core(loc_gpu_id + numGPUsLocal);
    cudaSetDevice(loc_gpu_id);
    /* if (loc_gpu_id == 0 || true) {
      std::cout << "Output thread begins for device: " << loc_gpu_id << std::endl;
    } */
    for (unsigned job_id = 0; job_id < dm_jobs.size(); job_id++) {
      /* if (loc_gpu_id == 0 || true) {
        std::cout << "GPU: " << loc_gpu_id << ", out DM job: " << job_id << std::endl;
      }  */
      auto &dm_job = dm_jobs[job_id];

      // Wait for DtoH copy to finish for this job
      dm_job.cpu_lock.lock(); // Makes sure that dm_job.outputEnd record is called from host before waiting for it
      dm_job.outputEnd.synchronize(); // Wait for d2h copy to finish

      // Info
#ifdef DEDISP_DEBUG
      if (enable_verbose_iteration_reporting) {
        std::cout << "Copy output " << dm_job.idm_start << " to "
                  << dm_job.idm_end << " with " << dm_job.ndm_current << " ndms"
                  << std::endl;
      }
#endif
      

      dedisp_size dst_stride = cmd.fftoutP ? 1ULL * nsamp_padded * out_bytes_per_sample : 1ULL * nsamps_computed * out_bytes_per_sample;

      if (local_gpu_id == 0 ) {
        //std::cout << "GPU: " << loc_gpu_id << ", out DM job: " << job_id << " begin memcpy" << std::endl;
        
        // copy part from pinned h_data_t_dm to part of paged return buffer out
        // GPU Host mem pointers
        dedisp_size src_stride = 1ULL * nsamp_padded * out_bytes_per_sample;
        float* h_src_float = (float*)dm_job.h_data_t_dm->data();
        auto *h_src = (void*) h_src_float;
        // CPU mem pointers

        dedisp_size dst_offset = 1ULL * dm_job.idm_start * dst_stride;
        auto *h_dst = (void *)(((size_t)out) + dst_offset);
        mCopyMem.start();
        memcpy2D_internal(h_dst,               // dst
                dst_stride,          // dst stride
                h_src,               // src
                src_stride,          // src stride
                dst_stride,          // width bytes
                dm_job.ndm_current); // height
        mCopyMem.end();
        //std::cout << "GPU: " << loc_gpu_id << ", out DM job: " << job_id << " end memcpy" << std::endl;
      }

      // Signal that the host buffer can be used again
      if ((job_id + ndm_buffers) < ndm_jobs) {
        dm_jobs[job_id + ndm_buffers].gpu_lock.unlock();
      }
    }
    /* if (loc_gpu_id == 0 || true) {
      std::cout << "Output thread ends for device: " << loc_gpu_id << std::endl;
    } */
  });
  #endif
  


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

  //std::cout << " GPU: " << local_gpu_id << ", channels: " << start_chan << " - " << end_chan << std::endl;
  // Process all dm batches (outer dm jobs)
  for (unsigned dm_job_id_outer = 0; dm_job_id_outer < dm_jobs.size(); dm_job_id_outer += ndm_buffers) 
  {
    //std::cout << " GPU: " << local_gpu_id << ", dm_job_id_outer = " << dm_job_id_outer << std::endl;
    // Process all channel batches
    for (unsigned channel_job_id = 0; channel_job_id < channel_jobs.size(); channel_job_id++) 
    {
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
      if (channel_job_id == 0) 
      {

        // This can probably be optimized further if the input is pinned memory
        // The two steps can be reduced to one using cudaMemcpy2DAsync, directly
        // copying from in to d_in_ptr, skipping the host buffer.

        // "in" is local to the node. Globally, it begins at start_chan_node
        dedisp_size gulp_chan_byte_idx =
            ((channel_job.ichan_start - start_chan_node) / chans_per_word) * sizeof(dedisp_word);

        /* memcpy2D_internal(channel_job.h_in_ptr,    // dst
                 dst_stride,              // dst width
                 in + gulp_chan_byte_idx, // src. 
                 src_stride,              // src width
                 dst_stride,              // width bytes (represents how many columns actually copied?)
                 nsamp);                  // height */

        htodstream->record(channel_job.inputStart);
        /* htodstream->memcpyHtoDAsync(channel_job.d_in_ptr, // dst
                                    channel_job.h_in_ptr, // src
                                    nsamp * nchan_words_gulp * sizeof(dedisp_float));  // size */

        htodstream->memcpyHtoD2DAsync(channel_job.d_in_ptr, // dst
                                       dst_stride,          // dst pitch
                                       in + gulp_chan_byte_idx, // src
                                       src_stride,          // src pitch
                                       dst_stride,          // width bytes
                                       nsamp);              // height
        htodstream->record(channel_job.inputEnd);
      }
      executestream->waitEvent(channel_job.inputEnd);

      // Modified transpose_unpack kernel to just transpose floats
      executestream->record(channel_job.preprocessingStart);

      transpose_unpack((float *)channel_job.d_in_ptr,       // d_in
                       nchan_words_gulp,                    // input width
                       nsamp,                               // input height
                       nchan_words_gulp,                    // in_stride
                       nsamp_padded,                        // out_stride
                       d_data_x_nu,                         // d_out
                       in_nbits, 32,                        // in_nbits, out_nbits
                       1.0 / nchan,                         // scale
                       *executestream);                     // stream

      // Apply zero padding
      auto dst_ptr = ((float *)d_data_x_nu.data()) + nsamp;
      unsigned int nsamp_padding = nsamp_padded - nsamp;
      cu::checkError(cudaMemset2DAsync(dst_ptr,                       // devPtr
                                       nsamp_padded * sizeof(float),  // pitch
                                       0,                             // value
                                       nsamp_padding * sizeof(float), // width
                                       nchan_batch_max,               // height
                                       *executestream));
      
      // FFT data (real to complex) along time axis
      for (unsigned int i = 0; i < nchan_batch_max / nchan_fft_batch; i++) 
      {
        cufftReal *idata = (cufftReal *)d_data_x_nu.data() +
                           i * nsamp_padded * nchan_fft_batch;
        cufftComplex *odata = (cufftComplex *)idata;
        R2Ctimer.Start();
        cufftResult result = cufftExecR2C(plan_r2c, idata, odata);
        R2Ctimer.Stop();
        r2ctime += R2Ctimer.Elapsed();
        if (result != CUFFT_SUCCESS) 
        {
            throw std::runtime_error("Error creating real to complex FFT plan.");
        }
      }
      executestream->record(channel_job.preprocessingEnd);

      // Process DM batches (inner dm jobs)
      for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++) 
      {
        unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
        if (dm_job_id >= dm_jobs.size()) 
          break;
        
        auto &dm_job = dm_jobs[dm_job_id];

        #ifdef DEDISP_DEBUG
                // Info
                std::cout << "Processing DM " << dm_job.idm_start << " to "
                          << dm_job.idm_end << std::endl;

        #endif

        // Initialize output to zero
        if (channel_job_id == 0) 
        {
          // Wait for previous output copy to finish
          if (dm_job_id_outer > 0) 
          {
            auto &dm_job_previous = dm_jobs[dm_job_id - ndm_buffers];
            // Done so that the dm buffer can be safely zeroed
            //nvtxRangePushA("DM job prev. outputEnd");  
            m2.start();
            dm_job_previous.outputEnd.synchronize();
            m2.end();
            //nvtxRangePop();
          }

          dm_job.d_data_x_dm->zero(*executestream);
        }

        // Commenting this out because it seems that chan_job.outputEnd is never recorded
        // Wait for temporary output from previous job to be copied
        /* if (channel_job_id > (nchan_buffers - 1)) 
        {
          auto &chan_job_previous = channel_jobs[channel_job_id - nchan_buffers];
          chan_job_previous.outputEnd.synchronize();
        } */

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
      if (channel_job_id_next < channel_jobs.size()) 
      {
        auto &channel_job_next = channel_jobs[channel_job_id_next];
        dedisp_size gulp_chan_byte_idx =
            ((channel_job_next.ichan_start - start_chan_node) / chans_per_word) *
            sizeof(dedisp_word);

        /* memcpy2D_internal(channel_job_next.h_in_ptr, // dst
                 dst_stride,                // dst width
                 in + gulp_chan_byte_idx,   // src
                 src_stride,                // src width
                 dst_stride,                // width bytes
                 nsamp);                    // height */

        htodstream->record(channel_job_next.inputStart);

        htodstream->memcpyHtoD2DAsync(channel_job_next.d_in_ptr, // dst
                                       dst_stride,          // dst pitch
                                       in + gulp_chan_byte_idx, // src
                                       src_stride,          // src pitch
                                       dst_stride,          // width bytes
                                       nsamp);              // height

        /* htodstream->memcpyHtoDAsync(channel_job_next.d_in_ptr, // dst
                                    channel_job_next.h_in_ptr, // src
                                    nsamp * nchan_words_gulp * sizeof(dedisp_float));  */      // size
        htodstream->record(channel_job_next.inputEnd);
      }

      // Is this synchronize really necessary here? 
      // - It is because of the single d_data_x_nu buffer through which all jobs pass 
      // Wait for current batch to finish
      //nvtxRangePushA("Finishing #ndm_buffer dedispersion kernels, moving to next channel batch");
      m1.start();
      executestream->synchronize();
      m1.end();
      //nvtxRangePop();

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
    for (unsigned dm_job_id_inner = 0; dm_job_id_inner < ndm_buffers; dm_job_id_inner++) 
    {
      unsigned dm_job_id = dm_job_id_outer + dm_job_id_inner;
      if (dm_job_id >= dm_jobs.size()) 
        break;

      auto &dm_job = dm_jobs[dm_job_id];

      #ifdef DEDISP_DEBUG
            // Info

            std::cout << "Post-processing DM " << dm_job.idm_start << " to "
                      << dm_job.idm_end << " with job_id " << dm_job_id << std::endl;

      #endif

      // Get pointer to DM output data on host and on device
      dedisp_size dm_stride = 1ULL * nsamp_padded * out_bytes_per_sample;
      
      auto *d_out = (float *)dm_job.d_data_x_dm->data();

      // Fourier transform results back to time domain if required
      executestream->record(dm_job.postprocessingStart);

      if (!cmd.fftoutP) 
      {
        for (unsigned int i = 0; i < ndm_batch_max / ndm_fft_batch; i++) {
          cufftReal *odata =
              (cufftReal *)d_out + i * nsamp_padded * ndm_fft_batch;
          cufftComplex *idata = (cufftComplex *)odata;
          C2Rtimer.Start();
          cufftResult result = cufftExecC2R(plan_c2r, idata, odata);
          C2Rtimer.Stop();
          c2rtime += C2Rtimer.Elapsed();
          if (result != CUFFT_SUCCESS) {
              throw std::runtime_error("Error creating real to complex FFT plan.");
          }
        }

        // FFT scaling
        kernel.scale(dm_job.ndm_current, // height
                      nsamp_padded,       // width
                      nsamp_padded,       // stride
                      1.0f / nsamp_fft,   // scale
                      d_out,              // d_data
                      *executestream);    // stream
      } // end if !cmd.fftoutP

      // Reducing the dedispersed timeseries on all GPUs (all nodes) using NCCL
      
      // Don't need to use the group semantics here because
      // GPUs are handled by different threads

      // TODO: Try MPI reduce instead of NCCL and compare performance

      // sendbuf = recvbuf => in-place operation
      NCCLCHECK(ncclReduce((const void*) d_out, d_out, 
                              dm_job.ndm_current * nsamp_padded, 
                              ncclFloat, ncclSum, 0,
                              comm, 
                              *executestream // Different streams can be used to do reductions in parallel!?
                            ));

      executestream->record(dm_job.postprocessingEnd);

      #ifdef USECUFILE
      // Writing out the data directly from the GPU buffer
      // The final timeseries resides on GPU 0 on node 0
      std::string outname = "dm_chunk_" + std::to_string(dm_job_id_outer + dm_job_id_inner) + ".dat";
      if (local_gpu_id == 0 && start_chan_node == 0) {
        gpuDirectWrite(outname, d_out, round_up_4k(sizeof_data_x_dm));
      }
      
      // When beginning to write, make sure that the GPU buffer 
      // is locked until write is complete
      executestream->synchronize();

      #else
      // Copy output. If inverse FFT is not applied, the fourier coefficients are copied
      // Output is picked up by (already running) host side thread
      // and is there copied from CPU pinned to paged memory
      dm_job.gpu_lock.lock();
      dtohstream->waitEvent(dm_job.postprocessingEnd); // Ensures d2h begins only after device buffer is ready
      dtohstream->record(dm_job.outputStart);
      if (local_gpu_id == 0) {
        auto *h_out = dm_job.h_data_t_dm->data();
        dedisp_size size = 1ULL * dm_job.ndm_current * dm_stride;
        dtohstream->memcpyDtoHAsync(h_out, // dst
                                    d_out, // src
                                    size); // size
      }
      
      dtohstream->record(dm_job.outputEnd);
      dm_job.cpu_lock.unlock(); // Ensures that the output thread is able to wait for dm_job.outputEnd only after calling the above function
      #endif
    
    } // end for dm_job_id_inner
  } // end for dm_job_id_outer

  /* if (local_gpu_id == 0 || true) {
    std::cout << "Scheduled work for all DMS on device: " << local_gpu_id << std::endl;
  } */

  #ifndef USECUFILE
  // Wait for final memory transfer
  // Wait for host threads to exit
  if (output_thread.joinable()) {
    output_thread.join();
  }
  #endif

  /* if (local_gpu_id == 0) {
    std::cout << "Output thread joined for device: " << local_gpu_id << std::endl;
  } */
  
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
  cufftDestroy(plan_c2r);
  cufftDestroy(plan_r2c);
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
  cufftHandle plan_r2c, plan_c2r;
  int n[] = {(int)nfft};
  std::thread thread_r2c = std::thread([&]() {
    int inembed[] = {(int)nsamp_good};
    int onembed[] = {(int)nfreq_chunk_padded};
    cufftResult result =
        cufftPlanMany(&plan_r2c,              // plan
                      1, n,                   // rank, n
                      inembed, 1, inembed[0], // inembed, istride, idist
                      onembed, 1, onembed[0], // onembed, ostride, odist
                      CUFFT_R2C,              // type
                      nchunk);                // batch
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Error creating real to complex FFT plan.");
    }
    cufftSetStream(plan_r2c, *executestream);
  });
  std::thread thread_c2r = std::thread([&]() {
    int inembed[] = {(int)nfreq_chunk_padded};
    int onembed[] = {(int)nfreq_chunk_padded * 2};
    cufftResult result =
        cufftPlanMany(&plan_c2r,              // plan
                      1, n,                   // rank, n
                      inembed, 1, inembed[0], // inembed, istride, idist
                      onembed, 1, onembed[0], // onembed, ostride, odist
                      CUFFT_C2R,              // type
                      nchunk);                // batch
    if (result != CUFFT_SUCCESS) {
      throw std::runtime_error("Error creating complex to real FFT plan.");
    }
    cufftSetStream(plan_c2r, *executestream);
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
      cu::checkError(cudaMemset2DAsync(dst_ptr,                       // devPtr
                                       nsamp_padded * sizeof(float),  // pitch
                                       0,                             // value
                                       nsamp_padding * sizeof(float), // width
                                       nchan_batch_max,               // height
                                       *executestream));

      // FFT data (real to complex) along time axis
      for (unsigned int ichan = 0; ichan < channel_job.nchan_current; ichan++) {
        auto *idata =
            (cufftReal *)d_data_t_nu.data() + (1ULL * ichan * nsamp_padded);
        auto *odata = (cufftComplex *)d_data_f_nu.data() +
                      (1ULL * ichan * nsamp_padded / 2);
        cufftExecR2C(plan_r2c, idata, odata);
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
            (cufftComplex *)d_data_f_dm_ptr + (1ULL * idm * nsamp_padded / 2);
        auto *odata =
            (cufftReal *)d_data_t_dm_ptr + (1ULL * idm * nsamp_padded);
        cufftExecC2R(plan_c2r, idata, odata);
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
  cufftDestroy(plan_c2r);
  cufftDestroy(plan_r2c);
}

// Private helper function
void FDDGPUPlan::generate_spin_frequency_table(dedisp_size nfreq,
                                               dedisp_size nsamp_fft,
                                               dedisp_float dt) {
  h_spin_frequencies.resize(nfreq);

  int genspinfreq_cores = total_cores / numGPUsLocal - 1;

  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    pin_thread_to_core(numGPUsLocal + local_gpu_id * genspinfreq_cores + thread_id);

    #pragma omp for
    for (unsigned int ifreq = 0; ifreq < nfreq; ifreq++) {
      // Skipping the scaling of 1/dt in the spin frequencies which saves a multiplication
      // in the GPU kernel
      h_spin_frequencies[ifreq] = ifreq * (1.0 / (nsamp_fft));
    }
  }


  d_spin_frequencies.resize(nfreq * sizeof(dedisp_float));

  htodstream->memcpyHtoDAsync(d_spin_frequencies, h_spin_frequencies.data(),
                              d_spin_frequencies.size());
}

} // end namespace dedisp