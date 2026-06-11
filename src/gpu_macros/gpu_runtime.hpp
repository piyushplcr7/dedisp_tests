#ifndef GPU_RUNTIME_H
#define GPU_RUNTIME_H

#include <stdio.h>
#include <stdlib.h>

// I first define the error handling macro and related definitions. I will
// then use those to wrap all other macros, so that error handling is done
// automatically when using "gpu*" calls.

#ifdef USE_CUDA
    #include <cuda_runtime.h>
    #include <device_launch_parameters.h>
    #include <texture_fetch_functions.h>
    #define gpuError_t cudaError_t
    #define gpuSuccess cudaSuccess
    #define gpuGetErrorString cudaGetErrorString
    #define gpuHostRegisterPortable cudaHostRegisterPortable
    #define gpuEvent_t cudaEvent_t
    #define gpuStreamNonBlocking cudaStreamNonBlocking
    #define gpuHostAllocPortable cudaHostAllocPortable
    #define gpuEventDefault cudaEventDefault
    #define gpuTextureObject_t cudaTextureObject_t
#endif

#ifdef USE_HIP
    #include <hip/hip_runtime.h>
    #define gpuError_t hipError_t
    #define gpuSuccess hipSuccess
    #define gpuGetErrorString hipGetErrorString
    #define gpuHostRegisterPortable 0
    #define gpuEvent_t hipEvent_t
    #define gpuStreamNonBlocking hipStreamNonBlocking
    #define gpuHostAllocPortable hipHostMallocPortable
    #define gpuEventDefault hipEventDefault
    #define gpuTextureObject_t hipTextureObject_t
#endif

#if defined(USE_CUDA) || defined(USE_HIP)
inline void gpu_check_error(gpuError_t x, const char *file, int line){
    if(x != gpuSuccess){
        fprintf(stderr, "GPU error (%s:%d): %s\n", file, line, gpuGetErrorString(x));
        exit(1);
    }
}
#define GPU_CHECK_ERROR(X) \
    do { gpu_check_error((X), __FILE__, __LINE__); } while (0)
#endif // USE_CUDA || USE_HIP

#ifdef USE_CUDA

#define gpuHostAllocDefault cudaHostAllocDefault
#define gpuMalloc(...) GPU_CHECK_ERROR(cudaMalloc(__VA_ARGS__))
#define gpuHostAlloc(...) GPU_CHECK_ERROR(cudaHostAlloc(__VA_ARGS__)) // 3 args
#define gpuMallocHost(...) GPU_CHECK_ERROR(cudaMallocHost(__VA_ARGS__)) // 2 args
#define gpuMemcpy(...) GPU_CHECK_ERROR(cudaMemcpy(__VA_ARGS__))
#define gpuMemcpyAsync(...) GPU_CHECK_ERROR(cudaMemcpyAsync(__VA_ARGS__))
#define gpuMemset(...) GPU_CHECK_ERROR(cudaMemset(__VA_ARGS__))
#define gpuDeviceSynchronize() GPU_CHECK_ERROR(cudaDeviceSynchronize())
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuFree(...) GPU_CHECK_ERROR(cudaFree(__VA_ARGS__))
#define gpuHostFree(...) GPU_CHECK_ERROR(cudaFreeHost(__VA_ARGS__))
#define gpuStream_t cudaStream_t
#define gpuStreamCreate(...) GPU_CHECK_ERROR(cudaStreamCreate(__VA_ARGS__))
#define gpuStreamDestroy(...) GPU_CHECK_ERROR(cudaStreamDestroy(__VA_ARGS__))
#define gpuEventCreate(...) GPU_CHECK_ERROR(cudaEventCreate(__VA_ARGS__))
#define gpuEventCreateWithFlags(...) GPU_CHECK_ERROR(cudaEventCreateWithFlags(__VA_ARGS__))
#define gpuGetDeviceCount(...) GPU_CHECK_ERROR(cudaGetDeviceCount(__VA_ARGS__))
#define gpuGetLastError cudaGetLastError
#define gpuGetDevice(...) GPU_CHECK_ERROR(cudaGetDevice(__VA_ARGS__))
#define gpuSetDevice(...) GPU_CHECK_ERROR(cudaSetDevice(__VA_ARGS__))
#define gpuDeviceGetAttribute(...) GPU_CHECK_ERROR(cudaDeviceGetAttribute(__VA_ARGS__))
#define gpuDeviceAttributeWarpSize cudaDevAttrWarpSize
#define gpu_shfl_down(...) __shfl_down_sync(0xffffffff, __VA_ARGS__)
#define gpuMemGetInfo(...) GPU_CHECK_ERROR(cudaMemGetInfo(__VA_ARGS__))
#define gpuGetDeviceProperties(...) GPU_CHECK_ERROR(cudaGetDeviceProperties(__VA_ARGS__))
#define gpuDeviceProp_t cudaDeviceProp
#define gpuPeekAtLastError cudaPeekAtLastError
#define gpuHostRegister(...) GPU_CHECK_ERROR(cudaHostRegister(__VA_ARGS__))
#define gpuHostUnregister(...) GPU_CHECK_ERROR(cudaHostUnregister(__VA_ARGS__))
#define gpuEventDestroy(...) GPU_CHECK_ERROR(cudaEventDestroy(__VA_ARGS__))
#define gpuEventRecord(...) GPU_CHECK_ERROR(cudaEventRecord(__VA_ARGS__))
#define gpuEventSynchronize(...) GPU_CHECK_ERROR(cudaEventSynchronize(__VA_ARGS__))
#define gpuEventElapsedTime(...) GPU_CHECK_ERROR(cudaEventElapsedTime(__VA_ARGS__))
#define gpuStreamSynchronize(...) GPU_CHECK_ERROR(cudaStreamSynchronize(__VA_ARGS__))
#define gpuMemsetAsync(...) GPU_CHECK_ERROR(cudaMemsetAsync(__VA_ARGS__))
#define gpuStreamCreateWithFlags(...) GPU_CHECK_ERROR(cudaStreamCreateWithFlags(__VA_ARGS__))
#define gpuMemcpy2DAsync(...) GPU_CHECK_ERROR(cudaMemcpy2DAsync(__VA_ARGS__))
#define gpuStreamWaitEvent(...) GPU_CHECK_ERROR(cudaStreamWaitEvent(__VA_ARGS__))
#define gpuMemcpyToSymbolAsync(...) GPU_CHECK_ERROR(cudaMemcpyToSymbolAsync(__VA_ARGS__))
#define gpuSymbol(x) (x)
#define gpuCreateChannelDesc cudaCreateChannelDesc
#define gpuMallocArray(...) GPU_CHECK_ERROR(cudaMallocArray(__VA_ARGS__))
#define gpuMemcpyToArray(...) GPU_CHECK_ERROR(cudaMemcpyToArray(__VA_ARGS__))
#define gpuMemcpy2DToArray(...) GPU_CHECK_ERROR(cudaMemcpy2DToArray(__VA_ARGS__))
#define gpuArray_t cudaArray_t
#define gpuResourceDesc cudaResourceDesc
#define gpuResourceTypeArray cudaResourceTypeArray
#define gpuTextureDesc cudaTextureDesc
#define gpuAddressModeClamp cudaAddressModeClamp
#define gpuFilterModePoint cudaFilterModePoint
#define gpuReadModeElementType cudaReadModeElementType
#define gpuCreateTextureObject(...) GPU_CHECK_ERROR(cudaCreateTextureObject(__VA_ARGS__))
#define gpuBindTexture(...) GPU_CHECK_ERROR(cudaBindTexture(__VA_ARGS__))
#define gpuMemset2DAsync(...) GPU_CHECK_ERROR(cudaMemset2DAsync(__VA_ARGS__))
#define gpuMemcpyHostToHost cudaMemcpyHostToHost
#define gpuChannelFormatDesc cudaChannelFormatDesc

// Complex number operations:
#define gpuCreal cuCreal
#define gpuCimag cuCimag
#define gpuCadd  cuCadd
#define gpuCmul  cuCmul
#define gpuCdiv  cuCdiv
#define gpuConj  cuConj
#define gpuCsub  cuCsub
#define gpuCabs  cuCabs
#define gpuDoubleComplex cuDoubleComplex
#define gpuFloatComplex cuFloatComplex
#define make_gpuDoubleComplex make_cuDoubleComplex
#define make_gpuFloatComplex make_cuFloatComplex

#endif // USE_CUDA


#ifdef USE_HIP

#define gpuHostAllocDefault hipHostMallocDefault
#define gpuMalloc(...) GPU_CHECK_ERROR(hipMalloc(__VA_ARGS__))
#define gpuHostAlloc(...) GPU_CHECK_ERROR(hipHostMalloc(__VA_ARGS__)) // 3 args in hip match cuda
#define gpuMallocHost(...) GPU_CHECK_ERROR(hipHostMalloc(__VA_ARGS__, hipHostMallocDefault)) // 2 args in cuda, 3 required for hip -> manual injection
#define gpuMemcpy(...) GPU_CHECK_ERROR(hipMemcpy(__VA_ARGS__))
#define gpuMemcpyAsync(...) GPU_CHECK_ERROR(hipMemcpyAsync(__VA_ARGS__))
#define gpuMemset(...) GPU_CHECK_ERROR(hipMemset(__VA_ARGS__))
#define gpuDeviceSynchronize() GPU_CHECK_ERROR(hipDeviceSynchronize())
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuFree(...) GPU_CHECK_ERROR(hipFree(__VA_ARGS__))
#define gpuHostFree(...) GPU_CHECK_ERROR(hipHostFree(__VA_ARGS__))
#define gpuStream_t hipStream_t
#define gpuStreamCreate(...) GPU_CHECK_ERROR(hipStreamCreate(__VA_ARGS__))
#define gpuStreamDestroy(...) GPU_CHECK_ERROR(hipStreamDestroy(__VA_ARGS__))
#define gpuEventCreate(...) GPU_CHECK_ERROR(hipEventCreate(__VA_ARGS__))
#define gpuEventCreateWithFlags(...) GPU_CHECK_ERROR(hipEventCreateWithFlags(__VA_ARGS__))
#define gpuGetDeviceCount(...) GPU_CHECK_ERROR(hipGetDeviceCount(__VA_ARGS__))
#define gpuGetLastError hipGetLastError
#define gpuGetDevice(...) GPU_CHECK_ERROR(hipGetDevice(__VA_ARGS__))
#define gpuSetDevice(...) GPU_CHECK_ERROR(hipSetDevice(__VA_ARGS__))
#define gpuDeviceGetAttribute(...) GPU_CHECK_ERROR(hipDeviceGetAttribute(__VA_ARGS__))
#define gpuDeviceAttributeWarpSize hipDeviceAttributeWarpSize
#define gpu_shfl_down(...) __shfl_down(__VA_ARGS__)
#define gpuMemGetInfo(...) GPU_CHECK_ERROR(hipMemGetInfo(__VA_ARGS__))
#define gpuGetDeviceProperties(...) GPU_CHECK_ERROR( hipGetDeviceProperties(__VA_ARGS__) )
#define gpuDeviceProp_t hipDeviceProp_t
#define gpuPeekAtLastError hipPeekAtLastError
#define gpuHostRegister(...) GPU_CHECK_ERROR(hipHostRegister(__VA_ARGS__))
#define gpuHostUnregister(...) GPU_CHECK_ERROR(hipHostUnregister(__VA_ARGS__))
#define gpuEventDestroy(...) GPU_CHECK_ERROR(hipEventDestroy(__VA_ARGS__))
#define gpuEventRecord(...) GPU_CHECK_ERROR(hipEventRecord(__VA_ARGS__))
#define gpuEventSynchronize(...) GPU_CHECK_ERROR(hipEventSynchronize(__VA_ARGS__))
#define gpuEventElapsedTime(...) GPU_CHECK_ERROR(hipEventElapsedTime(__VA_ARGS__))
#define gpuStreamSynchronize(...) GPU_CHECK_ERROR(hipStreamSynchronize(__VA_ARGS__))
#define gpuMemsetAsync(...) GPU_CHECK_ERROR(hipMemsetAsync(__VA_ARGS__))
#define gpuStreamCreateWithFlags(...) GPU_CHECK_ERROR(hipStreamCreateWithFlags(__VA_ARGS__))
#define gpuMemcpy2DAsync(...) GPU_CHECK_ERROR(hipMemcpy2DAsync(__VA_ARGS__))
#define gpuStreamWaitEvent(...) GPU_CHECK_ERROR(hipStreamWaitEvent(__VA_ARGS__))
#define gpuMemcpyToSymbolAsync(...) GPU_CHECK_ERROR(hipMemcpyToSymbolAsync(__VA_ARGS__))
#define gpuSymbol(x) HIP_SYMBOL(x)
#define gpuCreateChannelDesc hipCreateChannelDesc
#define gpuMallocArray(...) GPU_CHECK_ERROR(hipMallocArray(__VA_ARGS__))
#define gpuMemcpyToArray(...) GPU_CHECK_ERROR(hipMemcpyToArray(__VA_ARGS__))
#define gpuMemcpy2DToArray(...) GPU_CHECK_ERROR(hipMemcpy2DToArray(__VA_ARGS__))
#define gpuArray_t hipArray_t
#define gpuResourceDesc hipResourceDesc
#define gpuResourceTypeArray hipResourceTypeArray
#define gpuTextureDesc hipTextureDesc
#define gpuAddressModeClamp hipAddressModeClamp
#define gpuFilterModePoint hipFilterModePoint
#define gpuReadModeElementType hipReadModeElementType
#define gpuCreateTextureObject(...) GPU_CHECK_ERROR(hipCreateTextureObject(__VA_ARGS__))
#define gpuBindTexture(...) GPU_CHECK_ERROR(hipBindTexture(__VA_ARGS__))
#define gpuMemset2DAsync(...) GPU_CHECK_ERROR(hipMemset2DAsync(__VA_ARGS__))
#define gpuMemcpyHostToHost hipMemcpyHostToHost
#define gpuChannelFormatDesc hipChannelFormatDesc

// Complex number operations:
#define gpuCreal hipCreal
#define gpuCimag hipCimag
#define gpuCadd  hipCadd
#define gpuCmul  hipCmul
#define gpuCdiv  hipCdiv
#define gpuConj  hipConj
#define gpuCsub  hipCsub
#define gpuCabs  hipCabs
#define gpuDoubleComplex hipDoubleComplex
#define gpuFloatComplex  hipFloatComplex
#define make_gpuDoubleComplex make_hipDoubleComplex
#define make_gpuFloatComplex make_hipFloatComplex
#endif

#ifdef USE_OPENMP
    #include <cstring>        // memcpy / memset
    #include <cstdlib>        // malloc / free
    #include <chrono>
    #include <sys/sysinfo.h>  // sysinfo() for host memory query

    // ---- types ----
    typedef int gpuError_t;
    #define gpuSuccess 0
    typedef int gpuEvent_t;
    typedef int gpuStream_t;
    struct gpuDeviceProp_t { int major; int minor; size_t totalConstMem; };
    // flag constants referenced by cu:: classes / drivers
    #define gpuStreamNonBlocking 0
    #define gpuEventDefault 0
    #define gpuHostAllocPortable 0
    #define gpuHostAllocDefault 0
    #define gpuMemcpyHostToDevice 0
    #define gpuMemcpyDeviceToHost 0
    #define gpuMemcpyDeviceToDevice 0
    #define gpuMemcpyHostToHost 0

    // ---- memory ----
    #define gpuMalloc(ptr, size)        (*(ptr) = std::malloc(size))
    #define gpuHostAlloc(ptr, size, fl) (*(ptr) = std::malloc(size))
    #define gpuMallocHost(ptr, size)    (*(ptr) = std::malloc(size))
    #define gpuFree(ptr)                std::free(ptr)
    #define gpuHostFree(ptr)            std::free(ptr)
    #define gpuHostRegister(...)        ((void)0)
    #define gpuHostUnregister(...)      ((void)0)

    // ---- copies (async == sync here) ----
    #define gpuMemcpy(dst, src, n, kind)            std::memcpy((dst), (src), (n))
    #define gpuMemcpyAsync(dst, src, n, kind, strm) std::memcpy((dst), (src), (n))
    #define gpuMemset(ptr, val, n)                  std::memset((ptr), (val), (n))
    #define gpuMemsetAsync(ptr, val, n, strm)       std::memset((ptr), (val), (n))

    // strided 2D copies: (dst, dpitch, src, spitch, widthBytes, height, [kind], [strm])
    static inline void gpu_omp_memcpy2d(void* dst, size_t dpitch, const void* src,
                                        size_t spitch, size_t widthBytes, size_t height) {
        for (size_t r = 0; r < height; ++r)
            std::memcpy((char*)dst + r*dpitch, (const char*)src + r*spitch, widthBytes);
    }
    #define gpuMemcpy2DAsync(dst, dpitch, src, spitch, w, h, kind, strm) \
        gpu_omp_memcpy2d((dst), (dpitch), (src), (spitch), (w), (h))
    static inline void gpu_omp_memset2d(void* dst, size_t dpitch, int val,
                                        size_t widthBytes, size_t height) {
        for (size_t r = 0; r < height; ++r)
            std::memset((char*)dst + r*dpitch, val, widthBytes);
    }
    #define gpuMemset2DAsync(dst, dpitch, val, w, h, strm) \
        gpu_omp_memset2d((dst), (dpitch), (val), (w), (h))

    // ---- streams / events: all synchronous no-ops ----
    #define gpuStreamCreate(s)            ((void)0)
    #define gpuStreamCreateWithFlags(...) ((void)0)
    #define gpuStreamDestroy(s)           ((void)0)
    #define gpuStreamSynchronize(s)       ((void)0)
    #define gpuStreamWaitEvent(...)       ((void)0)
    #define gpuEventCreate(e)             ((void)0)
    #define gpuEventCreateWithFlags(...)  ((void)0)
    #define gpuEventDestroy(e)            ((void)0)
    #define gpuEventRecord(e, s)          ((void)0)
    #define gpuEventSynchronize(e)        ((void)0)
    // cu::Event::elapsedTime is not on the FDD timing path (aa_gpu_timer is); 0 is fine.
    #define gpuEventElapsedTime(ms, a, b) (*(ms) = 0.0f)
    #define gpuDeviceSynchronize()        ((void)0)

    // ---- device management ----
    #define gpuGetDeviceCount(n)   (*(n) = 1)
    #define gpuSetDevice(d)        ((void)0)
    #define gpuGetDevice(d)        (*(d) = 0)
    #define gpuGetLastError()      (0)
    #define gpuPeekAtLastError()   (0)
    // used by cu::CU.cpp error/device helpers (compiled into the `cu` lib)
    #define gpuGetErrorString(x)               ("openmp")
    #define gpuGetDeviceProperties(props, dev) ((void)0)
    // On the CPU backend "device memory" is host RAM (gpuMalloc == host malloc),
    // so report real host free/total. This lets the FDD planner's ndm_buffers
    // sizing loop self-tune to a single outer DM pass when RAM allows, instead
    // of being pinned to one buffer by a fake zero-memory device.
    static inline int gpu_omp_meminfo(size_t* freep, size_t* totalp) {
        struct sysinfo si;
        if (sysinfo(&si) != 0) {           // fall back to a large value on failure
            if (totalp) *totalp = (size_t)1 << 40;
            if (freep)  *freep  = (size_t)1 << 40;
            return 0;
        }
        if (totalp) *totalp = (size_t)si.totalram * si.mem_unit;
        if (freep)  *freep  = (size_t)si.freeram  * si.mem_unit;
        return 0;
    }
    #define gpuMemGetInfo(freep, totalp)       gpu_omp_meminfo((freep), (totalp))
    #define GPU_CHECK_ERROR(X)                 ((void)0)

    // ---- constant-memory symbol copy (used by FDDKernel::copy_delay_table) ----
    // Under OPENMP the "symbol" is a real host array; copy into it directly.
    #define gpuSymbol(x) (&(x))
    #define gpuMemcpyToSymbolAsync(sym, src, n, off, kind, strm) \
        std::memcpy((char*)(sym) + (off), (src), (n))

    // ---- warp shuffle (used in reductions): no-op identity on a single "lane" ----
    #define gpu_shfl_down(val, delta) (val)
#endif

#define gpuCheckLastError() GPU_CHECK_ERROR(gpuGetLastError())

#endif