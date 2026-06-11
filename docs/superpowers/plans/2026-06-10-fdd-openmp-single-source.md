# FDD OpenMP single-source mirror — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the same `src/fdd/FDDGPUPlan.cpp` + its kernel files compile and run under `-DGPU_BACKEND=OPENMP` (plain host OpenMP, NextSilicon-compiler-friendly), replacing the CUDA runtime/cuFFT/4 kernels with host equivalents; retire `FDDCPUPlan`.

**Architecture:** Approach A — each kernel file keeps its CUDA body under `#if defined(USE_CUDA)||defined(USE_HIP)` and gains a sibling `#ifdef USE_OPENMP` host body (`#pragma omp parallel for`) with the *same wrapper signature*. All async/stream/event machinery is neutralized to synchronous no-ops in the `gpu_*` macro headers, so `FDDGPUPlan.cpp` compiles unchanged. cuFFT → FFTW.

**Tech Stack:** C++20, OpenMP (host), FFTW3 (`fftw3f`), MPI, CMake. Spec: `docs/superpowers/specs/2026-06-10-fdd-openmp-single-source-design.md`.

**Working constraints (read first):**
- **The user runs ALL builds and runs** on kuma (module env + conda `prestoenv_new`). Do NOT build in the agent sandbox. Every "verify" step below is a command for the **user** to run; hand them exact commands.
- **Never `git add`/`git commit`.** The user controls all staging/commits. This plan therefore has NO commit steps; replace "commit" checkpoints with "user reviews diff".
- There is no unit-test framework in this repo. The acceptance test is a build + run + `xcorr_shift.py` comparison against the GPU reference (lag 0, corr 1.0).
- Standard build command the user runs (reference):
  `cmake -S . -B build_omp -DGPU_BACKEND=OPENMP && cmake --build build_omp -j`

---

## File Structure

| File | Change | Responsibility under USE_OPENMP |
|---|---|---|
| `src/gpu_macros/gpu_runtime.hpp` | extend | Host types + malloc/memcpy/no-op stream-event macros incl. `gpuMemcpyToSymbolAsync`, `gpu_shfl_down`, 2D copies |
| `src/gpu_macros/gpu_tracer_tools.hpp` | extend | NVTX symbols → no-ops |
| `src/gpu_macros/gpu_asm.hpp` | extend | `gpu_fmul_rn`/`gpu_fma_rn_ftz`/`gpu_sin`/`gpu_cos` → host math |
| `src/gpu_macros/gpu_fft.hpp` | extend | `gpufft*` → FFTW3 (`fftwf_*`) |
| `src/gpu_macros/gpu_xccl.hpp` | extend | no include (no NCCL anywhere) |
| `src/gpu_macros/gpu_vectypes.hpp` | **create** | host `float2`/`make_float2` + complex helpers for the kernels |
| `src/fdd/helper.h` | extend | `aa_gpu_timer` host (`std::chrono`) version |
| `src/fdd/unpack/unpack_kernel.cuh` | guard | CUDA kernels under CUDA/HIP only |
| `src/fdd/unpack/unpack.cu` | add branch | `transpose_unpack` host body |
| `src/fdd/dedisperse/fdd_kernel.cuh` | guard + add | CUDA bits under CUDA/HIP; host `c_delay_table` + helpers |
| `src/fdd/dedisperse/FDDKernel.cu` | add branch | `launch`/`scale`/`copy_delay_table` host bodies |
| `src/fdd/CMakeLists.txt` | rewrite OPENMP branch | build `fdd` from `FDDGPUPlan.cpp` + kernels (as CXX) + `cu` + FFTW + OpenMP + MPI; drop `FDDCPUPlan` |
| `bin/test/CMakeLists.txt` | edit | build `testdedisp_new` under OPENMP; drop `testdedisp_omp` |
| `src/fdd/FDDCPUPlan.{cpp,hpp}`, `bin/test/testdedisp_omp.cpp` | **delete** (final task) | retired |

---

## Task 1: Host runtime macros (`gpu_runtime.hpp`)

**Files:**
- Modify: `src/gpu_macros/gpu_runtime.hpp` (the `#ifdef USE_OPENMP` block at lines 204-207)

- [ ] **Step 1: Replace the stub USE_OPENMP block with a full host shim**

Replace lines 204-207 (`#ifdef USE_OPENMP ... #endif`) with:

```cpp
#ifdef USE_OPENMP
    #include <cstring>   // memcpy / memset
    #include <cstdlib>   // malloc / free
    #include <chrono>

    // ---- types ----
    typedef int gpuError_t;
    #define gpuSuccess 0
    typedef struct { std::chrono::high_resolution_clock::time_point t; } gpuEvent_t;
    typedef int gpuStream_t;
    typedef int gpuDeviceProp_t;
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
    #define gpuEventRecord(e, s)          ((e)->t = std::chrono::high_resolution_clock::now())
    #define gpuEventSynchronize(e)        ((void)0)
    static inline void gpu_omp_elapsed(float* ms, gpuEvent_t a, gpuEvent_t b) {
        *ms = std::chrono::duration<float, std::milli>(b.t - a.t).count();
    }
    #define gpuEventElapsedTime(ms, a, b) gpu_omp_elapsed((ms), (a), (b))
    #define gpuDeviceSynchronize()        ((void)0)

    // ---- device management ----
    #define gpuGetDeviceCount(n)   (*(n) = 1)
    #define gpuSetDevice(d)        ((void)0)
    #define gpuGetDevice(d)        (*(d) = 0)
    #define gpuGetLastError()      (0)
    #define gpuPeekAtLastError()   (0)

    // ---- constant-memory symbol copy (used by FDDKernel::copy_delay_table) ----
    // Under OPENMP the "symbol" is a real host array; copy into it directly.
    #define gpuSymbol(x) (&(x))
    #define gpuMemcpyToSymbolAsync(sym, src, n, off, kind, strm) \
        std::memcpy((char*)(sym) + (off), (src), (n))

    // ---- warp shuffle (used in reductions): no-op identity on a single "lane" ----
    #define gpu_shfl_down(val, delta) (val)
#endif
```

- [ ] **Step 2: Keep the existing `gpuCheckLastError()` line (after the block) intact.** Confirm `gpuGetLastError` is defined for OPENMP (it is, above).

- [ ] **Step 3 (user verify, optional now):** This header is exercised by later build tasks; no standalone check.

- [ ] **Step 4: User reviews the diff.**

> Note: `cudaMemcpyToSymbol` signature is `(symbol, src, count, offset, kind)`. The repo's macro passes `(gpuSymbol(c_delay_table), src, count, offset, kind, stream)` (6 args incl. stream) — the OPENMP macro above matches that 6-arg form used at `FDDKernel.cu:17`.

---

## Task 2: Tracer, asm, xccl shims

**Files:**
- Modify: `src/gpu_macros/gpu_tracer_tools.hpp`
- Modify: `src/gpu_macros/gpu_asm.hpp`
- Modify: `src/gpu_macros/gpu_xccl.hpp`

- [ ] **Step 1: `gpu_tracer_tools.hpp` — add before the final `#endif`:**

```cpp
#ifdef USE_OPENMP
  typedef int gpuEventAttributes_t;
  typedef int gpuRangeId_t;
  #define gpuRangePushA(msg)  (0)
  #define gpuRangePop()       (0)
  #define gpuMarkA(msg)       ((void)0)
#endif
```

- [ ] **Step 2: `gpu_asm.hpp` — add before the final `#endif`:**

```cpp
#ifdef USE_OPENMP
  #include <cmath>
  #define gpu_fmul_rn(out, a, b)        ((out) = (a) * (b))
  #define gpu_fma_rn_ftz(out, a, b, c)  ((out) = std::fma((a), (b), (c)))
  #define gpu_sin(r, a)                 ((r) = std::sin((a)))
  #define gpu_cos(r, a)                 ((r) = std::cos((a)))
#endif
```

> Rationale: the GPU uses `sin.approx`/`cos.approx` SFU intrinsics; on host we use `std::sin/std::cos`. This is the documented source of the ~1e-3 residual vs the GPU reference and is acceptable per the spec.

- [ ] **Step 3: `gpu_xccl.hpp` — add before the final `#endif`:**

```cpp
#ifdef USE_OPENMP
  // No collective-library include: there are no nccl*/ncclComm* call sites
  // anywhere in fdd/ or common/. The MPI path uses MPI_* directly.
#endif
```

- [ ] **Step 4: User reviews the diff.**

---

## Task 3: Host `float2` types + `aa_gpu_timer`

**Files:**
- Create: `src/gpu_macros/gpu_vectypes.hpp`
- Modify: `src/fdd/helper.h` (the `aa_gpu_timer` struct, currently guarded `#if defined(USE_CUDA)||defined(USE_HIP)` at lines 10-36)

- [ ] **Step 1: Create `src/gpu_macros/gpu_vectypes.hpp`:**

```cpp
#ifndef GPU_VECTYPES_HPP
#define GPU_VECTYPES_HPP
// Host stand-ins for CUDA vector types, used by the FDD kernels under USE_OPENMP.
#ifdef USE_OPENMP
struct float2 { float x, y; };
static inline float2 make_float2(float x, float y) { return float2{x, y}; }
#endif
#endif // GPU_VECTYPES_HPP
```

- [ ] **Step 2: Make the OpenMP kernel files see it.** At the top of `src/fdd/dedisperse/fdd_kernel.cuh` (after the existing `#include "gpu_asm.hpp"` on line 3) and at the top of `src/fdd/unpack/unpack_kernel.cuh`, add:

```cpp
#include "gpu_vectypes.hpp"
```

- [ ] **Step 3: Add an OpenMP `aa_gpu_timer` in `src/fdd/helper.h`.** The existing struct is wrapped in `#if defined(USE_CUDA) || defined(USE_HIP) ... #endif` (lines 10-36). Immediately after that block's `#endif // USE_CUDA || USE_HIP`, add:

```cpp
#ifdef USE_OPENMP
#include <chrono>
struct aa_gpu_timer {
    std::chrono::high_resolution_clock::time_point t0, t1;
    void Start() { t0 = std::chrono::high_resolution_clock::now(); }
    void Stop()  { t1 = std::chrono::high_resolution_clock::now(); }
    float Elapsed() { return std::chrono::duration<float>(t1 - t0).count(); }
};
#endif // USE_OPENMP
```

- [ ] **Step 4: User reviews the diff.**

---

## Task 4: FFTW mapping (`gpu_fft.hpp`)

**Files:**
- Modify: `src/gpu_macros/gpu_fft.hpp` (add a `USE_OPENMP` branch before the final `#endif`)

**Context:** `FDDGPUPlan.cpp` uses (CUDA names): `gpufftHandle`, `gpufftPlanMany`, `gpufftSetStream`, `gpufftExecR2C`, `gpufftExecC2R`, `gpufftDestroy`, types `gpufftReal`/`gpufftComplex`, dir consts `GPUFFT_R2C`/`GPUFFT_C2R`. Confirm exact call sites first:
Run: `grep -n "gpufft" src/fdd/FDDGPUPlan.cpp`

- [ ] **Step 1: Add the OPENMP FFTW branch:**

```cpp
#ifdef USE_OPENMP
  #include <fftw3.h>
  typedef float        gpufftReal;
  typedef fftwf_complex gpufftComplex;
  #define GPUFFT_R2C 0
  #define GPUFFT_C2R 1
  #define GPUFFT_SUCCESS 0
  typedef int gpufftResult;

  // Handle stores the dims so Exec can build/execute a many-plan. We rebuild the
  // plan from saved dims at Exec time against the actual buffers (FFTW plans are
  // pointer-specific); FFTW_ESTIMATE keeps planning cheap.
  struct gpufftHandle_t {
      int type;        // GPUFFT_R2C or GPUFFT_C2R
      int n;           // logical transform length
      int batch;
      int istride, idist, ostride, odist;
  };
  typedef gpufftHandle_t gpufftHandle;

  // cufftPlanMany(&plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type, batch)
  static inline gpufftResult gpufftPlanMany(
      gpufftHandle* plan, int rank, int* n,
      int* inembed, int istride, int idist,
      int* onembed, int ostride, int odist,
      int type, int batch) {
      (void)rank; (void)inembed; (void)onembed;
      *plan = gpufftHandle{ type, n[0], batch, istride, idist, ostride, odist };
      return GPUFFT_SUCCESS;
  }
  static inline gpufftResult gpufftPlan1d(gpufftHandle* plan, int n, int type, int batch) {
      *plan = gpufftHandle{ type, n, batch, 1, n, 1, (type==GPUFFT_R2C ? n/2+1 : n) };
      return GPUFFT_SUCCESS;
  }
  #define gpufftSetStream(plan, stream) GPUFFT_SUCCESS

  static inline gpufftResult gpufftExecR2C(gpufftHandle p, gpufftReal* in, gpufftComplex* out) {
      int n = p.n;
      fftwf_plan fp = fftwf_plan_many_dft_r2c(
          1, &n, p.batch,
          in,  nullptr, p.istride, p.idist,
          out, nullptr, p.ostride, p.odist,
          FFTW_ESTIMATE);
      fftwf_execute(fp);
      fftwf_destroy_plan(fp);
      return GPUFFT_SUCCESS;
  }
  static inline gpufftResult gpufftExecC2R(gpufftHandle p, gpufftComplex* in, gpufftReal* out) {
      int n = p.n;
      fftwf_plan fp = fftwf_plan_many_dft_c2r(
          1, &n, p.batch,
          in,  nullptr, p.istride, p.idist,
          out, nullptr, p.ostride, p.odist,
          FFTW_ESTIMATE);
      fftwf_execute(fp);
      fftwf_destroy_plan(fp);
      return GPUFFT_SUCCESS;
  }
  #define gpufftDestroy(plan) GPUFFT_SUCCESS
#endif
```

- [ ] **Step 2: Reconcile the exact `gpufftExec*` call signature** found in Step-1's grep against these wrappers (cuFFT `cufftExecR2C(plan, in, out)` is by value for the handle; the macro form in `gpu_fft.hpp` for CUDA wraps it in `GPUFFT_CHECK`). If `FDDGPUPlan.cpp` calls `gpufftExecR2C(plan, a, b)` expecting a checked macro, wrap these inline fns in a `GPUFFT_CHECK(...)`-compatible expression, mirroring the CUDA defines (lines 24-25). Adjust stride/dist mapping to match the cuFFT `gpufftPlanMany` arguments actually used in the plan (verify against the call site).

- [ ] **Step 3: User reviews the diff and confirms the FFTW plan args mirror the cuFFT call** (istride/idist/ostride/odist/batch). Cross-check against the working FFTW calls in the (still-present) `FDDCPUPlan.cpp` `fft_r2c_inplace`/`fft_c2r` for the correct in-place R2C layout (`nsamp_padded = 2*(nsamp_fft/2+1)`, `idist = nsamp_padded`, `odist = nsamp_fft/2+1`).

---

## Task 5: Translate the unpack/transpose kernel

**Files:**
- Modify: `src/fdd/unpack/unpack_kernel.cuh` (guard CUDA kernels)
- Modify: `src/fdd/unpack/unpack.cu` (add host body to `transpose_unpack`)

- [ ] **Step 1: Guard the CUDA kernels in `unpack_kernel.cuh`.** Wrap the two `__global__` kernels (`transpose_kernel`, `transpose_unpack_kernel`, lines 7-162) in:

```cpp
#if defined(USE_CUDA) || defined(USE_HIP)
// ... existing transpose_kernel and transpose_unpack_kernel unchanged ...
#endif
```

(The `#define TILE_DIM/BLOCK_ROWS/EXPANSION` at the top stay outside the guard — the host body reuses `EXPANSION`.)

- [ ] **Step 2: Add the host body to `transpose_unpack` in `unpack.cu`.** Wrap the existing tiled-launch body (lines 30-97) so the function becomes:

```cpp
void transpose_unpack(
    const float* d_in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    float* d_out,
    dedisp_size in_nbits, dedisp_size out_nbits,
    float scale,
    gpuStream_t stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    // ... existing tiled launch body unchanged (lines 31-97) ...
#elif defined(USE_OPENMP)
    // out[y][x] = transpose of in[x][y]; here width = #input-rows-along-x,
    // height = #input-cols. Mirrors the kernels' index math:
    //   index_in  = x*in_stride + y     (input  [height_rows][in_stride])
    //   index_out = y*out_stride + x     (output [width_rows][out_stride])
    if (in_nbits == 32) {
        // Pure transpose, no offset/scale (matches transpose_kernel<float>).
        #pragma omp parallel for
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x)
                d_out[y * out_stride + x] = d_in[x * in_stride + y];
    } else if (in_nbits == 8) {
        // 8-bit packed words -> EXPANSION channels each, (val-127.5)*scale.
        // Matches transpose_unpack_kernel: input word grid is [height][width]
        // of dedisp_word; each word expands to EXPANSION output channels.
        const dedisp_word* in_w = reinterpret_cast<const dedisp_word*>(d_in);
        dedisp_size in_mask = (1u << in_nbits) - 1;
        #pragma omp parallel for
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                dedisp_word word = in_w[x * in_stride + y];
                for (unsigned j = 0; j < EXPANSION; ++j) {
                    dedisp_word val = (word >> (j * in_nbits)) & in_mask;
                    float result = (((float)val) - 127.5f) * scale;
                    // output channel = x*EXPANSION + j, time index = y
                    d_out[(x * EXPANSION + j) * out_stride + y] = result;
                }
            }
        }
    }
#endif
}
```

- [ ] **Step 3: Confirm `in_stride`/`out_stride`/`width`/`height` orientation** against the call site `FDDGPUPlan.cpp:1083` (`width=nchan_words_gulp`, `height=nsamp`, `out_stride=nsamp_padded`). For the 32-bit path this means: input is `[nsamp][nchan]` row-major (stride `in_stride`), output is `[nchan][nsamp_padded]` — i.e. `d_out[chan*out_stride + t] = d_in[t*in_stride + chan]`. Re-map the `x/y` naming in Step 2 if the grep shows the opposite orientation; the invariant to preserve is **`d_out[chan][t] = d_in[t][chan]`** for the 32-bit pure transpose (this is the orientation already validated in `FDDCPUPlan`).

- [ ] **Step 4: User reviews the diff.**

> The 32-bit pure transpose is the path actually exercised by the FITS driver (already validated at DM0 in the `FDDCPUPlan` work). The 8-bit branch mirrors `transpose_unpack_kernel` for completeness.

---

## Task 6: Translate the dedisperse + scale kernels

**Files:**
- Modify: `src/fdd/dedisperse/fdd_kernel.cuh` (guard CUDA; add host `c_delay_table`)
- Modify: `src/fdd/dedisperse/FDDKernel.cu` (host bodies for `launch`/`scale`/`copy_delay_table`)

- [ ] **Step 1: In `fdd_kernel.cuh`, make `c_delay_table` host-visible.** Replace line 8:

```cpp
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
```

with:

```cpp
#if defined(USE_CUDA) || defined(USE_HIP)
__constant__ dedisp_float c_delay_table[DEDISP_MAX_NCHANS];
#elif defined(USE_OPENMP)
dedisp_float c_delay_table[DEDISP_MAX_NCHANS];   // plain host array
#endif
```

- [ ] **Step 2: Guard the `__device__` helpers and the two `__global__` kernels** (operators, `cmac`, `raw_sin/raw_cos`, `dedisperse_kernel`, `scale_output_kernel`; lines 28-284) in:

```cpp
#if defined(USE_CUDA) || defined(USE_HIP)
// ... existing device helpers + kernels unchanged ...
#endif
```

The `#define NDM_BATCH_GRID/NFREQ_*` macros (lines 11-22) stay outside the guard.

- [ ] **Step 3: Add host bodies in `FDDKernel.cu`.** Make each method dual-bodied. `copy_delay_table`:

```cpp
void FDDKernel::copy_delay_table(const void* src, size_t count, size_t offset, gpuStream_t stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    gpuMemcpyToSymbolAsync(gpuSymbol(c_delay_table), src, count, offset,
                           gpuMemcpyDeviceToDevice, stream);
#elif defined(USE_OPENMP)
    std::memcpy((char*)gpuSymbol(c_delay_table) + offset, src, count);
#endif
}
```
(Add `#include <cstring>` and `#include <cmath>` near the top of `FDDKernel.cu`.)

- [ ] **Step 4: `FDDKernel::launch` host body** — replace the templated `CALL_KERNEL` switch (lines 48-86) so the function is:

```cpp
void FDDKernel::launch(
    dedisp_size ndm, dedisp_size nfreq, dedisp_size nchan, float dt,
    const dedisp_float* d_spin_frequencies, const dedisp_float* d_dm_list,
    const dedisp_float2* d_in, const dedisp_float2* d_out,
    dedisp_size in_stride, dedisp_size out_stride,
    unsigned int idm_start, unsigned int idm_end, unsigned int ichan_start,
    gpuStream_t stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    // ... existing grid/block + CALL_KERNEL switch unchanged ...
#elif defined(USE_OPENMP)
    // Host port of dedisperse_kernel<NCHAN,false>: freq-domain phase rotation,
    // ACCUMULATING onto d_out (the kernel reads-modifies-writes per channel gulp).
    const float* in  = reinterpret_cast<const float*>(d_in);   // float2 as [.,2]
    float*       out = reinterpret_cast<float*>(const_cast<dedisp_float2*>(d_out));
    #pragma omp parallel for collapse(2)
    for (unsigned idm = idm_start; idm < idm_end; ++idm) {
        for (size_t ifreq = 0; ifreq < nfreq; ++ifreq) {
            float dm = d_dm_list[idm];
            float f  = d_spin_frequencies[ifreq];
            size_t o = ((size_t)idm * out_stride + ifreq) * 2;
            float sx = out[o], sy = out[o + 1];
            for (unsigned ichan = 0; ichan < nchan; ++ichan) {
                size_t iidx = ((size_t)ichan * in_stride + ifreq) * 2;
                float bx = in[iidx], by = in[iidx + 1];
                float tdm   = roundf(dm * c_delay_table[ichan_start + ichan]);
                float phase = 2.0f * (float)M_PI * f * tdm;
                float c = cosf(phase), s = sinf(phase);
                // sum += sample * phasor  (complex)
                sx += bx * c - by * s;
                sy += bx * s + by * c;
            }
            out[o] = sx; out[o + 1] = sy;
        }
    }
#endif
}
```

> Note: `idm_end` here corresponds to the kernel's `idm_end`; the kernel indexes `d_dm_list[idm_start + idm_local]`, equivalent to `d_dm_list[idm]` for `idm` in `[idm_start, idm_end)`. `out_stride`/`in_stride` are in **float2 units** (hence `*2` for the float view). Confirm against the call site that `d_out` is pre-zeroed before the first channel gulp (the GPU relies on this; FDDGPUPlan zero-pads/inits `d_data_x_nu`).

- [ ] **Step 5: `FDDKernel::scale` host body** — replace lines 99-110:

```cpp
void FDDKernel::scale(
    dedisp_size height, dedisp_size width, dedisp_size stride,
    dedisp_float scale, dedisp_float* d_data, gpuStream_t stream)
{
#if defined(USE_CUDA) || defined(USE_HIP)
    dim3 grid(height); dim3 block(128);
    scale_output_kernel<<<grid, block, 0, stream>>>(width, stride, scale, d_data);
#elif defined(USE_OPENMP)
    #pragma omp parallel for
    for (size_t row = 0; row < height; ++row)
        for (size_t i = 0; i < width; ++i)
            d_data[row * stride + i] *= scale;   // matches scale_output_kernel
#endif
}
```

- [ ] **Step 6: User reviews the diff,** focusing on the complex-MAC sign convention (must equal `cmac`: `a.x += b.x*c.x - b.y*c.y`, `a.y += b.x*c.y + b.y*c.x`) and the float2-stride `*2`.

---

## Task 7: CMake — build the GPU plan under OPENMP

**Files:**
- Modify: `src/fdd/CMakeLists.txt` (the `if(GPU_BACKEND STREQUAL "OPENMP")` branch, lines 1-39)
- Modify: `bin/test/CMakeLists.txt`
- Check: `src/common/cuda/CMakeLists.txt` builds the `cu` target under OPENMP

**Context:** the top `CMakeLists.txt` already sets `GPU_LANG=CXX` and empty `GPU_FFT_LIBS`/`GPU_TRACER_LIBS`/`GPU_XCCL` interface libs under OPENMP, and the GPU branch already uses `set_source_files_properties(... LANGUAGE ${GPU_LANG})`. So the GPU recipe nearly works as-is for OPENMP.

- [ ] **Step 1: Replace the OPENMP branch body** (lines 2-38, the `fdd_cpu`/`FDDCPUPlan` target) with a target that mirrors the GPU `fdd` target but: (a) sources = `FDDGPUPlan.cpp` + kernels + `chunk.cpp` (NO `FDDCPUPlan.cpp`); (b) `.cu` set to `LANGUAGE CXX`; (c) links FFTW + OpenMP + MPI + `cu`:

```cmake
if(GPU_BACKEND STREQUAL "OPENMP")

    add_library(fdd_gpu_clig fdd_gpu.c)

    set_source_files_properties(
        dedisperse/FDDKernel.cu
        unpack/unpack.cu
        PROPERTIES LANGUAGE CXX
    )

    add_library(fdd SHARED
        FDDGPUPlan.cpp
        dedisperse/FDDKernel.cu
        unpack/unpack.cu
        chunk.cpp
        $<TARGET_OBJECTS:common>
        $<TARGET_OBJECTS:plan>
        $<TARGET_OBJECTS:external>
    )
    target_compile_definitions(fdd PRIVATE ${GPU_DEFINITIONS})  # USE_OPENMP
    target_include_directories(fdd
        PUBLIC  ${CMAKE_SOURCE_DIR}/src/gpu_macros
        PRIVATE ${CMAKE_SOURCE_DIR}/src
        PRIVATE ${CMAKE_SOURCE_DIR}/src/fdd
        PRIVATE ${CMAKE_SOURCE_DIR}/src/numa
        PRIVATE ${CMAKE_SOURCE_DIR}/src/dataHandlers
        PRIVATE ${CMAKE_SOURCE_DIR}/src/dataHandlers/fits
        PRIVATE ${CFITSIO_INCLUDE_DIR}
        PRIVATE ${MPI_CXX_INCLUDE_DIRS}
        PRIVATE ${FFTW3_INCLUDE_DIR}
    )
    target_link_libraries(fdd PRIVATE
        ${FFTW3F_LIBRARY}
        OpenMP::OpenMP_CXX
        cu
        fdd_gpu_clig
        hwlocutils
        fits
        ${CFITSIO_LIBRARY}
        datacontainer
        MPI::MPI_CXX
        GPU_TRACER_LIBS   # empty interface under OPENMP
        GPU_FFT_LIBS      # empty interface under OPENMP
        GPU_XCCL          # empty interface under OPENMP
    )
    set_target_properties(fdd PROPERTIES
        PUBLIC_HEADER FDDGPUPlan.hpp
        VERSION ${DEDISP_VERSION}
        SOVERSION ${DEDISP_VERSION_MAJOR}
        CXX_STANDARD 20
    )
    install(TARGETS fdd LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

else()
    # ===== existing GPU fdd library definition unchanged =====
    # (remove FDDCPUPlan.cpp from this target's sources too — see Step 4)
    ...
endif()
```

- [ ] **Step 2: Ensure the `cu` target compiles under OPENMP.** Open `src/common/cuda/CMakeLists.txt`; confirm it sets `LANGUAGE CXX`/`USE_OPENMP` under the OPENMP backend (add `set_source_files_properties(CU.cpp PROPERTIES LANGUAGE CXX)` and `target_compile_definitions(cu PRIVATE ${GPU_DEFINITIONS})` if missing, and link `${GPU_TRACER_LIBS}`). `CU.cpp` must be in the build of the `cu` library so `cu::` symbols resolve.

- [ ] **Step 3: `bin/test/CMakeLists.txt` — under OPENMP build `testdedisp_new`** linking `fdd` (instead of `testdedisp_omp` linking `fdd_cpu`). Mirror the GPU driver target but with `LANGUAGE CXX` and OpenMP/FFTW/MPI link. Verify which target currently builds under OPENMP:
Run: `grep -n "testdedisp\|fdd_cpu\|fdd\b\|OPENMP" bin/test/CMakeLists.txt`

- [ ] **Step 4: Do NOT touch the GPU (`else`) branch in this task.** Leave `FDDCPUPlan.cpp` in the GPU `fdd` target. The `dedisp` c-library (`src/clib`) still references `dedisp::FDDCPUPlan` (typeinfo + ctor/dtor), so removing it from the GPU build now breaks the GPU/CUDA link (`undefined reference to dedisp::FDDCPUPlan`). Retirement — removing `FDDCPUPlan.cpp` from the GPU target AND scrubbing every reference (clib, headers) — is done holistically in **Task 9**, after the OPENMP build validates. The OPENMP branch already excludes `FDDCPUPlan.cpp`, so the OPENMP build needs nothing here.

- [ ] **Step 5 (USER BUILD — first full compile):**
Run:
```bash
cmake -S . -B build_omp -DGPU_BACKEND=OPENMP
cmake --build build_omp -j 2>&1 | tee logs/build_omp.log
```
Expected: `fdd` and `testdedisp_new` link successfully. Iterate on compile errors (likely: a missing `gpu*` macro, an unguarded CUDA token, or an FFTW arg) — fix in the relevant Task 1-6 file, not here.

- [ ] **Step 6: User reviews the diff + build log.**

---

## Task 8: Validate against the GPU reference

**Files:** none (run + compare).

- [ ] **Step 1 (USER RUN):** run the OpenMP build on the standard FITS case (same args the GPU/`run_master.sh` uses), writing per-DM `.dat` to a fresh dir, e.g. `/scratch/panchal/ompout/`:
```bash
# (user adapts run_master.sh / openmp_test.sh to invoke build_omp/bin/testdedisp_new
#  with: -lodm 0 -dmstep 0.01 -numdms 1000 -multout -nobary, same FITS input)
```

- [ ] **Step 2 (USER VERIFY — acceptance gate):**
```bash
python3 ~/repos/dedisp_tests/xcorr_shift.py \
    /scratch/panchal/masterout/out_DM0.00.dat \
    /scratch/panchal/ompout/out_DM0.00.dat
```
Expected: **best lag = 0, normalized corr = 1.000000, max abs err ~1e-3.**
Repeat for a higher DM (`out_DM5.00.dat`) — also expect lag 0.

- [ ] **Step 3:** If lag ≠ 0 → the per-DM `dm_offset` in `FDDGPUPlan::writeOutput` (line 148, `get_max_delay()` vs `roundf(dm*delay[0])`) differs from the reference; align it (the per-DM form was the one validated in the prior session). If corr < 1 → a kernel-body sign/stride bug (revisit Task 5/6); use `compare_timeseries.py` to locate the first divergent sample.

- [ ] **Step 4: User confirms acceptance.**

---

## Task 9: Retire FDDCPUPlan

**Files:**
- Delete: `src/fdd/FDDCPUPlan.cpp`, `src/fdd/FDDCPUPlan.hpp`, `bin/test/testdedisp_omp.cpp`
- Grep-and-remove residual references.

- [ ] **Step 1:** Only after Task 8 passes. Find references:
Run: `grep -rn "FDDCPUPlan\|testdedisp_omp" src/ bin/ CMakeLists.txt`

- [ ] **Step 2:** Delete the three files and remove every remaining reference (CMake sources, includes). Re-run the OPENMP build (Task 7 Step 5) and a CUDA build if available to confirm nothing depended on them.

- [ ] **Step 3: User reviews the diff and the final OPENMP + (if possible) CUDA builds both pass.**

---

## Self-review notes (author)
- Spec coverage: backend select (T7), cu:: shim via macros (T1, T7 step 2), 5 macro headers (T1,T2,T4), 4 kernels (T5,T6), FFTW (T4), aa_gpu_timer (T3), nchan%32 caveat (carried in kernel math; inputs already padded upstream), validation (T8), retirement (T9). ✓
- Open risk carried into execution: exact `gpufftExec*`/`gpufftPlanMany` arg mapping (T4 steps 2-3) and `transpose_unpack` orientation (T5 step 3) — both have explicit verify-against-call-site steps.
- No git commits anywhere (per user rule); "commit" checkpoints replaced by "user reviews diff".
