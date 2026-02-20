#include <iostream>
#include "fdd_gpu.h"
#include "fitscontainer.hpp"
#include "fits.hpp"
#include <vector>
#include "dedisp_types.h"
#include <string>
#include "cufft_optimal_size.hpp"
#include "fdd/helper.h"
#include "FDDGPUPlan.hpp"

int main(int argc, char **argv) {
  std::cout << "=========================================================================\n\n";
  std::cout << "                     GPU Fourier Domain Dedispersion         " << std::endl;
  std::cout << "                            By Piyush Panchal         \n" << std::endl;
  // Parse the CLI args
  Cmdline *cmd = parseCmdline(argc, argv);
  int nfitsfiles = cmd->argc;
  char **fitsfiles = cmd->argv;

#ifdef TESTDEDISP_DEBUG
  showOptionValues();
  printf("No. of infiles = %d\n", nfitsfiles);
  for (int i = 0; i < nfitsfiles; ++i) {
    printf("input file %d: %s \n", i, fitsfiles[i]);
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

  std::vector<std::string> listFitsNames(nfitsfiles);
  for (int i = 0 ; i < nfitsfiles ; ++i) {
    listFitsNames[i] = std::string(fitsfiles[i]);
  }

  fitsLoader container(listFitsNames, 0, 1, cmd->downsamp);
  // Contiguous buffer for assembled data
  //container.allocContiguousAssemblyBuf(); 
  // Reduce directly to the assembly buffer
  // container.reduceInAssemblyBuf();

  std::cout << "------------------------ LOADING + REDUCING FITS ------------------------\n" << std::endl;
  size_t chunksize = HALF_MAX_CHUNKSIZE;
  int poln = 0;
  container.ldSeq(chunksize, poln);
  if (!cmd->nobaryP) {
    std::cout << "\n------------------- GENERATING BARYCENTER CORRECTIONS -------------------\n" << std::endl;
    container.barycenter();
  }

  // nsamps scales with nfitsfiles because Tobs is scaled already
  dedisp_float dm_tol = 1.25;
  dedisp_size in_nbits = 8;
  dedisp_size out_nbits = 32; 
  
  std::cout << "\n------------------------------ PLAN CREATION ---------------------------\n" << std::endl;
  dedisp::FDDGPUPlan plan(container, 0);

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
    std::cout << "Generating equispaced DM list using the provided step size"
              << std::endl;
  #endif
    plan.generate_dm_list_equispaced(cmd->lodm, cmd->dmstep, cmd->numdms);
  }

  // Set the output parameters for the plan, which also allocates the output buffer
  plan.setOutputParams(cmd->cleanoutP, cmd->fftoutP, cmd->multoutP, out_nbits);

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
  plan.writeOutput(cmd->outfile, cmd->dmstepW, !cmd->nobaryP, container.getInForOut());
  plan.writeInfs(cmd->outfile, container.getFitsVector()[0], container.nsampsLocal(), container.sampletime(), cmd->dmstepW);
  std::cout << "\n------------------------ DEDISPERSION SUCCESSFUL ------------------------\n\n" << std::endl; 

  return 0;
}