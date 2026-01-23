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


#endif