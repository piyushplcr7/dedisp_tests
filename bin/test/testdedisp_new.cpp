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
  // Plain MPI_Init only promises that the thread which called it may safely
  // call MPI -- but FDDGPUPlan::execute_gpu() moves every MPI_Sendrecv /
  // MPI_File_iwrite_at / MPI_Wait / MPI_File_write_at call onto a dedicated
  // background std::thread ("mpi_thread"), never the thread that inits MPI
  // here. That is undefined behavior under plain MPI_Init and is the
  // suspected root cause of the multi-node hangs (Cray MPICH/CXI: lost
  // completion, silent spin) and segfaults (OpenMPI/PSM2: crash inside
  // ompi_request_default_wait) seen on Daint and OzSTAR respectively.
  // Request MPI_THREAD_MULTIPLE explicitly and verify it was actually
  // granted -- do not assume the request succeeded.
  auto mpi_init_t0 = std::chrono::steady_clock::now();
  int mpi_thread_provided = MPI_THREAD_SINGLE;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_provided);
  auto mpi_init_t1 = std::chrono::steady_clock::now();
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_thread_provided < MPI_THREAD_MULTIPLE) {
    const char *level_name =
        mpi_thread_provided == MPI_THREAD_SINGLE     ? "MPI_THREAD_SINGLE" :
        mpi_thread_provided == MPI_THREAD_FUNNELED   ? "MPI_THREAD_FUNNELED" :
        mpi_thread_provided == MPI_THREAD_SERIALIZED ? "MPI_THREAD_SERIALIZED" :
                                                        "unknown";
    fprintf(stderr,
            "[rank %d/%d] WARNING: requested MPI_THREAD_MULTIPLE but only "
            "%s was provided -- FDDGPUPlan's background mpi_thread calls "
            "MPI from a non-init thread and is not safe at this level.\n",
            mpi_rank, mpi_size, level_name);
    fflush(stderr);
  }

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
  size_t chunksize = HALF_MAX_CHUNKSIZE/2;
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
  // Default DM width must be set before setOutputParams so the .dat files
  // (named with w_ during execute) pair with the .inf files from writeInfs
  if (cmd->dmstepW == 0)
    cmd->dmstepW = 2;
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

  //plan.writeOutput(cmd->outfile, cmd->dmstepW, !cmd->nobaryP, container.getInForOut());

  // Write infs using last MPI proc
  if (mpi_rank == mpi_size-1 )
    plan.writeInfs(cmd->outfile, container.getFirstFile(), container.nsampsGlobal(), container.sampletime(), cmd->dmstepW, !cmd->nobaryP, container.blotoa(), container.avgvoverc());

  // ---- Runtime reporting -------------------------------------------------
  // Two numbers, both max-over-ranks (the run is finished only when the
  // slowest rank is):
  //   SOLVER  MPI_Init exit -> here, i.e. the work, MPI startup/teardown out
  //   MAIN    process start -> after MPI_Finalize, the whole program
  // MPI_Init is a barrier, so ranks enter the timed region together and the
  // max is meaningful rather than a launch-skew artefact.
  //
  // These replace reconstructing the same numbers from nsys reports: the
  // profiler is still needed for per-call breakdowns, but not for runtime.
  auto solver_t1 = std::chrono::steady_clock::now();
  double solver_sec =
      std::chrono::duration<double>(solver_t1 - mpi_init_t1).count();
  double init_sec =
      std::chrono::duration<double>(mpi_init_t1 - mpi_init_t0).count();

  double solver_max = 0.0, init_max = 0.0;
  MPI_Reduce(&solver_sec, &solver_max, 1, MPI_DOUBLE, MPI_MAX, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&init_sec, &init_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  MPI_Finalize();

  // After MPI_Finalize no collective is legal, so `main` is rank 0's own
  // whole-program time. It differs from the true max only by the spread in
  // MPI_Finalize, which is small next to the run itself.
  auto main_t1 = std::chrono::steady_clock::now();
  double main_sec =
      std::chrono::duration<double>(main_t1 - mpi_init_t0).count();
  double finalize_sec =
      std::chrono::duration<double>(main_t1 - solver_t1).count();

  if (mpi_rank == 0) {
    // solver_max and init_max are over all ranks; main and finalize are
    // rank 0's own -- do not subtract one from the other, they are not on
    // the same rank.
    printf("RUNTIME ranks=%d solver_max=%.3f main_rank0=%.3f "
           "init_max=%.3f finalize_rank0=%.3f\n",
           mpi_size, solver_max, main_sec, init_max, finalize_sec);
    fflush(stdout);
  }

  if (mpi_rank == 0)
    std::cout << "\n------------------------ DEDISPERSION SUCCESSFUL ------------------------\n\n" << std::endl; 

  return 0;
}