# FDD OpenMP single-source mirror — design

**Date:** 2026-06-10
**Branch:** `nextsilicon`
**Status:** approved design, pre-implementation

## Goal

Make the **same** `src/fdd/FDDGPUPlan.cpp` (and its kernel files) compile and run
under three backends selected by `-DGPU_BACKEND={CUDA,HIP,OPENMP}`. Under
`OPENMP` the file is compiled by an ordinary host C++ compiler (the target being
NextSilicon Maverick 2's own compiler, which auto-maps hot host-OpenMP loops to
its dataflow fabric), with the CUDA runtime, cuFFT, and the four CUDA kernels
replaced by host equivalents. The separate `FDDCPUPlan` reimplementation is
**retired**; its validated kernel math is relocated into the single-source
kernel files as the `USE_OPENMP` branch.

### Success criteria
- `cmake -DGPU_BACKEND=OPENMP` builds `FDDGPUPlan.cpp` + kernel TUs with the host
  C++ compiler and links FFTW + OpenMP + MPI.
- Running the existing FITS case under OPENMP produces per-DM `.dat` output that
  matches the GPU reference: `xcorr_shift.py <gpu_ref> <openmp_out>` reports
  **lag 0, normalized corr 1.0**, max abs err at float-noise level (~1e-3) at
  DM0 and at a higher DM.
- `FDDGPUPlan.cpp` itself is **unchanged** (call structure, async scaffolding,
  job/buffer orchestration all preserved); all backend differences live in the
  macro/shim headers, the kernel files, and CMake.

### Non-goals
- No algorithmic change to the GPU path. No performance tuning of the OpenMP
  path beyond `#pragma omp parallel for` on the kernel loops.
- No OpenMP *target offload* — plain host OpenMP only (upgradeable later).

## Chosen approach: A — guarded dual kernel bodies + backend shim headers

Each kernel translation unit keeps its CUDA kernel + launch under
`#if defined(USE_CUDA) || defined(USE_HIP)` and gains a sibling
`#ifdef USE_OPENMP` host implementation of the **same wrapper-function
signature**, written as `#pragma omp parallel for` loop nests reproducing the
per-output-element math. Because the wrapper signatures are identical across
backends, `FDDGPUPlan.cpp` calls them unchanged.

Rejected alternatives:
- **B (CUDA-emulation header):** `#define __global__/threadIdx/<<<>>>` → loops.
  The transpose kernels use `__shared__` tiles + `__syncthreads`, which do not
  emulate faithfully/efficiently on CPU.
- **C (shared per-element functor):** clean for the elementwise dedisperse/scale
  kernels but the tile/shared-mem transpose kernels do not fit a per-element
  functor. Possible later refinement, not the starting point.

## Components

### 1. Backend selection & build (`CMakeLists.txt`, `src/fdd/CMakeLists.txt`, `bin/test/CMakeLists.txt`)
- `GPU_BACKEND=OPENMP` already defines `USE_OPENMP`. Change the fdd CMake OPENMP
  branch so it **stops skipping** `FDDGPUPlan.cpp`, `dedisperse/FDDKernel.cu`,
  `unpack/unpack.cu`, and `common/cuda/CU.cpp`, and instead compiles them with
  the host C++ compiler via `set_source_files_properties(<.cu files> PROPERTIES
  LANGUAGE CXX)`.
- Link FFTW (`fftw3f`), OpenMP, MPI under OPENMP.
- Remove `FDDCPUPlan.{cpp,hpp}` and `bin/test/testdedisp_omp.cpp` from the build.
  The single driver is `bin/test/testdedisp_new.cpp` for all three backends
  (it already constructs `FDDGPUPlan` from the `dataLoader`). Its only CUDA-only
  dependency is **`aa_gpu_timer`** (`src/fdd/helper.h`, currently guarded
  `#if defined(USE_CUDA)||defined(USE_HIP)`): add a `USE_OPENMP` version backed
  by `std::chrono` exposing the same `Start()/Stop()/Elapsed()` interface. Its
  other calls (`gpuGetDeviceCount`→1, `gpuSetDevice/GetDevice`→no-op,
  `cu::Marker`) are covered by the shims.

### 2. `gpu_runtime.hpp` — `USE_OPENMP` runtime branch
Add a `USE_OPENMP` block defining everything `CU.cpp` and `FDDGPUPlan.cpp` use,
so both compile unchanged:
- Types: `gpuStream_t`, `gpuEvent_t` (small structs or `int`), `gpuError_t`,
  `gpuStream_t`/`gpuEvent_t` flag constants (`gpuStreamNonBlocking`,
  `gpuEventDefault`, `gpuHostAllocPortable`, `gpuHostAllocDefault`, the
  `gpuMemcpy*` direction enums).
- Memory: `gpuMalloc`/`gpuFree`/`gpuHostAlloc`/`gpuMallocHost`/`gpuHostFree`
  → `malloc`/`free`/aligned alloc.
- Copies: `gpuMemcpy`, `gpuMemcpyAsync` → `memcpy`; `gpuMemcpy2DAsync` → strided
  2D copy loop; `gpuMemset`/`gpuMemsetAsync`/`gpuMemset2DAsync` → `memset`/loop.
- Streams/events: `gpuStreamCreate/Destroy/Synchronize/WaitEvent`,
  `gpuEventCreate/Record/Synchronize/Destroy` → no-ops. `gpuEventElapsedTime`
  → 0.0f (optionally a `std::chrono` timestamp stored in the event struct for
  the benchmark prints).
- Device mgmt: `gpuGetDeviceCount`→1, `gpuSetDevice/GetDevice`→no-op,
  `gpuDeviceSynchronize`→no-op (already), `gpuMemGetInfo`→host RAM if needed.
- Net effect: the `cu::` wrapper classes in `CU.cpp` (Stream/Event/Memory/
  Marker), which are written against these macros, compile and run as
  synchronous host code with no rewrite.

### 3. `cu::` classes (`common/cuda/CU.{h,cpp}`)
No interface change. Only requirement is that `CU.cpp`'s method bodies resolve
all `gpu*` macros under OPENMP (provided by §2) and the NVTX symbols (provided
by §5). `DeviceMemory`/`HostMemory` allocate host memory; `Stream`/`Event`/
`Marker` become effectively no-ops. Verify `CU.cpp` has no raw CUDA tokens
outside the `gpu*` macro layer; guard any that exist.

### 4. `gpu_fft.hpp` — `USE_OPENMP` → FFTW
Add a `USE_OPENMP` branch mapping the `gpufft*` surface used by `FDDGPUPlan.cpp`
to FFTW single-precision:
- `gpufftHandle` → a struct holding an `fftwf_plan` (+ saved dims for re-exec).
- `gpufftPlanMany`/`gpufftPlan1d` → `fftwf_plan_many_dft_r2c`/`_c2r` with the
  batch/stride/dist arguments mapped from the cuFFT call.
- `gpufftExecR2C`/`gpufftExecC2R` → `fftwf_execute_dft_r2c`/`_c2r`.
- `gpufftSetStream` → no-op. `gpufftDestroy` → `fftwf_destroy_plan`.
- Reuse the FFTW plan/exec logic already proven in `FDDCPUPlan`
  (`fft_r2c_inplace` etc.) as the reference for argument mapping.

### 5. Remaining macro shims
- `gpu_asm.hpp` `USE_OPENMP`: `gpu_fmul_rn(o,a,b)`→`o=a*b`,
  `gpu_fma_rn_ftz(o,a,b,c)`→`o=fmaf(a,b,c)`, `gpu_sin/gpu_cos`→`sinf/cosf`.
- `gpu_tracer_tools.hpp` `USE_OPENMP`: `gpuRangePushA/Pop/MarkA` → no-ops;
  `gpuEventAttributes_t`/`gpuRangeId_t` → trivial typedefs (so `cu::Marker`
  compiles).
- `gpu_xccl.hpp` `USE_OPENMP`: no `#include`. **Confirmed:** there are no
  `nccl*`/`ncclComm*` call sites anywhere in `fdd/` or `common/` — the MPI path
  uses `MPI_*` directly (e.g. `MPI_Put`) — so guarding out the include is the
  only change.

### 6. Kernel translation (4 kernels)
Add `#ifdef USE_OPENMP` host bodies (plain `#pragma omp parallel for`) beside the
CUDA kernels, with bodies ported from the validated `FDDCPUPlan` equivalents:

| Wrapper (called by FDDGPUPlan) | File | OpenMP body source |
|---|---|---|
| `transpose_unpack(...)` (8-bit unpack + 32-bit pure transpose) | `unpack/unpack.cu`, `unpack/unpack_kernel.cuh` | `FDDCPUPlan` `transpose_data` (8-bit: `(val-127.5)*scale`; 32-bit: pure transpose) |
| `FDDKernel::launch(...)` (dedisperse, freq-domain phase mult) | `dedisperse/FDDKernel.cu`, `dedisperse/fdd_kernel.cuh:91` | `FDDCPUPlan` `dedisperse_optimized` |
| second dedisperse kernel (`fdd_kernel.cuh:273`, segmented variant) | same | `FDDCPUPlan` segmented dedisperse |
| `FDDKernel::scale(...)` (elementwise 1/nsamp_fft) | `dedisperse/FDDKernel.cu` | trivial scaled copy loop |
| `FDDKernel::copy_delay_table(...)` | `dedisperse/FDDKernel.cu` | `memcpy` into a host delay-table buffer |

Carry over the **`nchan` multiple-of-32 caveat**: kernels stride channels by 32
with no tail guard; pad channels with zeros so the last tile never reads OOB.

## Data flow (unchanged from GPU path)
FITS → `dataLoader` assembled float buffer (`in_nbits==32`) → `FDDGPUPlan::execute`
→ (per gulp) `transpose_unpack` → FFTW R2C → `FDDKernel::launch` (dedisperse) →
FFTW C2R → `FDDKernel::scale` → host output buffer → `writeOutput` per DM. Under
OPENMP all `cu::Stream`/`Event` ops are synchronous no-ops, so the existing
double-buffer/job-queue orchestration executes as straight-line code.

## Risks / open items
- **`.cu` as C++:** confirm `set_source_files_properties(LANGUAGE CXX)` is honored
  for `.cu` by the toolchain; fallback is thin `.cpp` TUs that `#include` the
  `.cu`. (Decide in plan.)
- **FFTW plan-arg mapping:** cuFFT advanced-layout args (istride/idist/ostride/
  odist/batch) must map exactly to `fftwf_plan_many_*`; mis-mapping silently
  corrupts output. Validate against `FDDCPUPlan`'s working FFTW calls.
- **Output-offset convention:** keep the per-DM `dm_offset` already adopted
  (`roundf(dm*delay[0])`) so OPENMP output aligns with the reference.

## Validation plan
1. Build under OPENMP on kuma (user runs the build).
2. Run the standard FITS case (`run` script), produce `out_DM0.00.dat` etc.
3. `xcorr_shift.py <gpu_ref> <openmp_out>` for DM0 and a higher DM → expect lag 0,
   corr 1.0, max err ~1e-3.
4. Spot-check the full all-DM output length matches the reference.

## Retirement
Once validation passes, delete `src/fdd/FDDCPUPlan.{cpp,hpp}` and
`bin/test/testdedisp_omp.cpp` and remove them from CMake. The kernel math they
contained now lives in the single-source kernel files under `USE_OPENMP`.

## Note on git
Per standing user instruction, this spec is **not** committed automatically; the
user controls all staging/commits.
