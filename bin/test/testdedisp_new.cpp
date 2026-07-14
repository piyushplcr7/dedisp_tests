#include <iostream>
#include <chrono>
#include <mpi.h>
#include "fdd_gpu.h"
#include "fitscontainer.hpp"
#include <vector>
#include "dedisp_types.h"
#include <string>
#include "cufft_optimal_size.hpp"
#include "fdd/helper.h"
#include "FDDGPUPlan.hpp"
#include "gpu_runtime.hpp"

int main(int argc, char **argv) {
  // Time MPI_Init with a non-MPI clock (MPI_Wtime is invalid before init).
  auto mpi_init_t0 = std::chrono::steady_clock::now();
  MPI_Init(&argc, &argv);
  auto mpi_init_t1 = std::chrono::steady_clock::now();
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

#ifdef TESTDEDISP_DEBUG
  double mpi_init_sec =
      std::chrono::duration<double>(mpi_init_t1 - mpi_init_t0).count();
  printf("[rank %d/%d] MPI_Init took %.3f sec\n", mpi_rank, mpi_size,
         mpi_init_sec);
  fflush(stdout);
#endif

  int gpu_device_id = 0;
  int num_gpus = 0;
  gpuGetDeviceCount(&num_gpus);
  const char *localid_str = std::getenv("SLURM_LOCALID");
  const char *ntasks_str  = std::getenv("SLURM_NTASKS_PER_NODE");
  if (localid_str && ntasks_str) {
    int local_id = std::atoi(localid_str);
    int ntasks   = std::atoi(ntasks_str);
    if (ntasks > 0)
      gpu_device_id = local_id * (num_gpus / ntasks);
  }
  gpuSetDevice(gpu_device_id);
  gpuGetDevice(&gpu_device_id);

#ifdef TESTDEDISP_DEBUG
  printf("[rank %d/%d] num_gpus=%d, SLURM_LOCALID=%s, SLURM_NTASKS_PER_NODE=%s, gpu device %d\n",
         mpi_rank, mpi_size, num_gpus,
         localid_str ? localid_str : "unset",
         ntasks_str ? ntasks_str : "unset",
         gpu_device_id);
  fflush(stdout);
#endif

  if (mpi_rank == 0) {
    std::cout << "=========================================================================\n\n";
    std::cout << "                     GPU Fourier Domain Dedispersion         " << std::endl;
    std::cout << "                            By Piyush Panchal         \n" << std::endl;
  }
  // Parse the CLI args
  Cmdline *cmd = parseCmdline(argc, argv);
  int nfiles = cmd->argc;
  char **fitsfiles = cmd->argv;

#ifdef TESTDEDISP_DEBUG
  if (mpi_rank == 0) {
    showOptionValues();
    printf("No. of infiles = %d\n", nfiles);
    for (int i = 0; i < nfiles; ++i) {
      printf("input file %d: %s \n", i, fitsfiles[i]);
    }
  }
#endif

  if (cmd->numdms > 0) {
    if (cmd->numdms < 256) {
      std::cerr << "ERROR: Please specify at least 256 numdms" << std::endl;
      exit(1);
    }
    if (cmd->dmstep == 0) {
      std::cerr
          << "ERROR: Non zero dmstep value required when specifying numdms!"
          << std::endl;
      exit(1);
    }
  }

  std::vector<std::string> listFileNames(nfiles);
  for (int i = 0 ; i < nfiles ; ++i) {
    listFileNames[i] = std::string(fitsfiles[i]);
  }

  // nsamps scales with nfiles because Tobs is scaled already
  dedisp_float dm_tol = 1.25;
  dedisp_size in_nbits = cmd->nbits;
  dedisp_size out_nbits = 32; 

  dataLoader container(listFileNames, mpi_rank, mpi_size, cmd->downsamp, in_nbits);

  // If nbits is not specified, use the nbits from the container (which is determined from the input files)
  in_nbits = in_nbits == 0 ? container.nbits() : in_nbits;

  if (mpi_rank == 0)
    std::cout << "------------------------ LOADING + REDUCING FITS ------------------------\n" << std::endl;
  size_t chunksize = HALF_MAX_CHUNKSIZE;
  int poln = 0;

  cu::Marker mFileRead("Reading fits files", cu::Marker::red);

  mFileRead.start();
  container.ldSeq(chunksize, poln);
  mFileRead.end();

  if (!cmd->nobaryP) {
    cu::Marker mBary("Generating barycentering correction", cu::Marker::green);

    mBary.start();

    if (mpi_rank == 0)
      std::cout << "\n------------------- GENERATING BARYCENTER CORRECTIONS -------------------\n" << std::endl;
    container.barycenter();
    mBary.end();
  }
  
  cu::Marker mPlan("Plan creation", cu::Marker::green);
  mPlan.start();

  if (mpi_rank == 0)
    std::cout << "\n------------------------------ PLAN CREATION ---------------------------\n" << std::endl;

  dedisp::FDDGPUPlan plan(container, gpu_device_id);
  mPlan.end();

  cu::Marker mDMlist("Gen DM list", cu::Marker::green);
  mDMlist.start();

  if (mpi_rank == 0)
    std::cout << "\n--------------------------- GENERATING DM LIST --------------------------\n" << std::endl;
  // Generate a list of dispersion measures for the plan
  if (cmd->numdms == 0) {
  #ifdef TESTDEDISP_DEBUG
    std::cout << "Numdms not specified, generating DM list using the internal "
                 "function"
              << std::endl;
  #endif
    plan.generate_dm_list(cmd->lodm, cmd->hidm, cmd->pwidth, dm_tol);
  } else {
  #ifdef TESTDEDISP_DEBUG
    if (mpi_rank == 0)
      std::cout << "Generating equispaced DM list using the provided step size"
                << std::endl;
  #endif
    plan.generate_dm_list_equispaced(cmd->lodm, cmd->dmstep, cmd->numdms);
  }
  mDMlist.end();

  cu::Marker mOutPar("Set output params", cu::Marker::green);
  mOutPar.start();
  // Set the output parameters for the plan, which also allocates the output buffer
  plan.setOutputParams(cmd->cleanoutP, cmd->fftoutP, cmd->multoutP, out_nbits, cmd->outfile, cmd->dmstepW, !cmd->nobaryP);
  mOutPar.end();

  if (mpi_rank == 0)
    std::cout << "\n------------------------------ PLAN EXECUTE -----------------------------\n" << std::endl;
  aa_gpu_timer timer;
  timer.Start();
  dedisp_byte *input = reinterpret_cast<dedisp_byte *> (container.getAssembledDataBfr().get());
  // Compute the dedispersion transform on the GPU
  plan.execute(container.nsampsLocal(), input, in_nbits, (dedisp_byte *)plan.output_buffer_.get(), out_nbits);
  timer.Stop();
  printf("plan.execute() took %.2f seconds\n", timer.Elapsed());

  if (cmd->dmstepW == 0)
    cmd->dmstepW = 2;
  //plan.writeOutput(cmd->outfile, cmd->dmstepW, !cmd->nobaryP, container.getInForOut());
  plan.writeInfs(cmd->outfile, container.getFileVector()[0].get(), container.nsampsLocal(), container.sampletime(), cmd->dmstepW, !cmd->nobaryP, container.blotoa(), container.avgvoverc());
  
  MPI_Finalize();

  if (mpi_rank == 0)
    std::cout << "\n------------------------ DEDISPERSION SUCCESSFUL ------------------------\n\n" << std::endl; 

  return 0;
}