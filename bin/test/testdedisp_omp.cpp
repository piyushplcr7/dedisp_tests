// CPU (OpenMP) + MPI driver: mirror of testdedisp_new.cpp but using FDDCPUPlan
// instead of FDDGPUPlan. Loads FITS via the MPI dataLoader, dedisperses on the
// CPU, and writes per-DM timeseries (.dat) + metadata (.inf) via multout.
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <mpi.h>
#include <vector>
#include <string>

#include "fdd_gpu.h"          // CLIG command-line parser (parseCmdline, Cmdline)
#include "fitscontainer.hpp"  // dataLoader
#include "dedisp_types.h"
#include "FDDCPUPlan.hpp"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int mpi_rank = 0, mpi_size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (mpi_rank == 0) {
    std::cout << "=========================================================================\n\n";
    std::cout << "             CPU (OpenMP) Fourier Domain Dedispersion         " << std::endl;
    std::cout << "                            By Piyush Panchal         \n" << std::endl;
  }

  // Parse the CLI args (same options as the GPU driver: -lodm -dmstep -numdms
  // -multout -cleanout -nobary -downsamp -o <basename> <fits files...>)
  Cmdline *cmd = parseCmdline(argc, argv);
  int nfiles = cmd->argc;
  char **fitsfiles = cmd->argv;

#ifdef TESTDEDISP_DEBUG
  showOptionValues();
  printf("No. of infiles = %d\n", nfiles);
  for (int i = 0; i < nfiles; ++i) {
    printf("input file %d: %s \n", i, fitsfiles[i]);
  }
#endif

  if (cmd->numdms > 0) {
    if (cmd->numdms < 256) {
      std::cerr << "ERROR: Please specify at least 256 numdms" << std::endl;
      exit(1);
    }
    if (cmd->dmstep == 0) {
      std::cerr << "ERROR: Non zero dmstep value required when specifying numdms!" << std::endl;
      exit(1);
    }
  }

  std::vector<std::string> listFileNames(nfiles);
  for (int i = 0 ; i < nfiles ; ++i) {
    listFileNames[i] = std::string(fitsfiles[i]);
  }

  dedisp_float dm_tol = 1.25;
  dedisp_size in_nbits = cmd->nbits;
  dedisp_size out_nbits = 32;

  dataLoader container(listFileNames, mpi_rank, mpi_size, cmd->downsamp, in_nbits);

  // If nbits is not specified, use the nbits from the container
  in_nbits = in_nbits == 0 ? container.nbits() : in_nbits;

  std::cout << "------------------------ LOADING + REDUCING FITS ------------------------\n" << std::endl;
  size_t chunksize = HALF_MAX_CHUNKSIZE;
  int poln = 0;
  container.ldSeq(chunksize, poln);

  if (!cmd->nobaryP) {
    std::cout << "\n------------------- GENERATING BARYCENTER CORRECTIONS -------------------\n" << std::endl;
    container.barycenter();
  }

  std::cout << "\n------------------------------ PLAN CREATION ---------------------------\n" << std::endl;
  dedisp::FDDCPUPlan plan(container, /*device_index=*/0);

  std::cout << "\n--------------------------- GENERATING DM LIST --------------------------\n" << std::endl;
  if (cmd->numdms == 0) {
    plan.generate_dm_list(cmd->lodm, cmd->hidm, cmd->pwidth, dm_tol);
  } else {
    plan.generate_dm_list_equispaced(cmd->lodm, cmd->dmstep, cmd->numdms);
  }

  // Set the output parameters for the plan, which also allocates output_buffer_
  plan.setOutputParams(cmd->cleanoutP, cmd->fftoutP, cmd->multoutP, out_nbits,
                       cmd->outfile, cmd->dmstepW, !cmd->nobaryP);

  std::cout << "\n------------------------------ PLAN EXECUTE -----------------------------\n" << std::endl;
  auto t0 = std::chrono::steady_clock::now();
  dedisp_byte *input = reinterpret_cast<dedisp_byte *>(container.getAssembledDataBfr().get());
  plan.execute(container.nsampsLocal(), input, in_nbits,
               (dedisp_byte *)plan.output_buffer_.get(), out_nbits);
  auto t1 = std::chrono::steady_clock::now();
  printf("plan.execute() took %.2f seconds\n",
         std::chrono::duration<double>(t1 - t0).count());

  if (cmd->dmstepW == 0)
    cmd->dmstepW = 2;

  // Write per-DM timeseries (.dat) and metadata (.inf)
  plan.writeOutput(cmd->outfile, cmd->dmstepW, !cmd->nobaryP, container.getInForOut());
  plan.writeInfs(cmd->outfile, container.getFileVector()[0].get(), container.nsampsLocal(),
                 container.sampletime(), cmd->dmstepW, !cmd->nobaryP,
                 container.blotoa(), container.avgvoverc());

  std::cout << "\n------------------------ DEDISPERSION SUCCESSFUL ------------------------\n\n" << std::endl;

  MPI_Finalize();
  return 0;
}
