#ifndef GPUXCCL_H
#define GPUXCCL_H

#ifdef USE_CUDA
  // NVIDIA / CUFFT :
  #include <nccl.h>
#endif

#ifdef USE_HIP
  // AMD / HIP :
  #include <rccl.h>

#endif

#ifdef USE_OPENMP
  // No collective-library include: there are no nccl*/ncclComm* call sites
  // anywhere in fdd/ or common/. The MPI path uses MPI_* directly.
#endif

#endif