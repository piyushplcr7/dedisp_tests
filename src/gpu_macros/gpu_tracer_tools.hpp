#ifndef GPU_TRACER_TOOLS_HPP
#define GPU_TRACER_TOOLS_HPP

#ifdef USE_CUDA
  // NVIDIA / CUFFT :
  #include <nvToolsExt.h>
  #define gpuRangePushA  nvtxRangePushA
  #define gpuRangePop    nvtxRangePop
  #define gpuMarkA       nvtxMarkA
  #define gpuEventAttributes_t nvtxEventAttributes_t
  #define gpuRangeId_t nvtxRangeId_t
#endif

#ifdef USE_HIP
  #include <roctx.h>
  #include <roctracer/roctracer.h>
  #include <roctracer/roctracer_ext.h>
  #include <roctracer/roctracer_hsa.h>

  #define gpuRangePushA  roctxRangePushA
  #define gpuRangePop    roctxRangePop
  #define gpuMarkA       roctxMarkA
  #define gpuEventAttributes_t roctracer_domain_t
  #define gpuRangeId_t uint32_t
#endif

#endif