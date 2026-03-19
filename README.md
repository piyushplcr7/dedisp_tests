# dedisp

A GPU-accelerated library for computing the incoherent dedispersion transform, a core operation in time-domain radio astronomy for pulsar and fast radio burst (FRB) detection.

This repository is forked from [svlugt/dedisp](https://github.com/svlugt/dedisp), which itself builds on Ben Barsdell's [original dedisp](https://code.google.com/p/dedisp) and [ajameson/dedisp](https://github.com/ajameson/dedisp). This fork extends the library with:

- **CUDA/HIP dual-backend support** for both NVIDIA and AMD GPUs
- **Multi-GPU and distributed computing** via MPI + NCCL/RCCL
- **FITS and filterbank file I/O** through an abstract data handler interface
- **Barycentric resampling** for pulsar timing corrections (using routines from PRESTO)
- **NUMA-aware memory management** via hwloc

For the original algorithm description, see [Barsdell et al. 2012](http://adsabs.harvard.edu/abs/2012arXiv1201.5380B).

## Dedispersion Implementations

The library provides three dedispersion algorithms sharing a common interface:

| Implementation | Class | Description | Performance |
|---|---|---|---|
| **Original** | `DedispPlan` | Texture memory, separate transpose + dedispersion | 1.0x (baseline) |
| **TDD** | `TDDPlan` | Combined unpack+transpose kernel, pinned memory, stream overlap | 1.5-2x |
| **FDD (GPU)** | `FDDGPUPlan` | FFT-based phase rotations, compute-bound | 2-2.4x (>512 DMs) |
| **FDD (CPU)** | `FDDCPUPlan` | FFTW + OpenMP reference, no GPU required | CPU-only |

- **TDD** improves on the original by overlapping data transfers with compute, using pinned memory, and fusing the unpack/transpose passes into a single kernel.
- **FDD** replaces time-domain shifts with phase rotations in the Fourier domain, reducing I/O and enabling better compute utilization. It scales better with future GPU architectures.
- **FDD CPU** is useful for validation without a GPU.

## Class Hierarchy

```
Plan                    -- abstract base: DM lists, delay tables, frequency metadata
└── GPUPlan             -- GPU base: device memory, streams, context
    ├── DedispPlan      -- original algorithm
    ├── TDDPlan         -- optimized time domain
    └── FDDGPUPlan      -- Fourier domain GPU
FDDCPUPlan              -- Fourier domain CPU reference (no GPU)
```

All implementations follow the same workflow: **create plan** -> **generate/set DM list** -> **execute**.

## Dependencies

| Dependency | Required | Purpose |
|---|---|---|
| CUDA 11.0.1+ or ROCm (HIP) | Yes | GPU backend |
| CMake 3.21+ | Yes | Build system |
| OpenMP | Yes | CPU parallelization |
| CFITSIO | Yes | FITS file I/O |
| FFTW3 | Yes | CPU FFT (FDD reference) |
| hwloc | Yes | NUMA topology awareness |
| MPI | Yes | Distributed computing |
| NCCL / RCCL | Yes | Multi-GPU collectives |
| libnuma | Optional | NUMA memory allocation |

Tested with GCC 8.3.0, CUDA 11.0.1, CMake 3.21+.

## Building

```bash
# Clone
git clone git@github.com:piyushplcr7/dedisp_tests.git
cd dedisp_tests

# Configure with CUDA backend
mkdir -p build && cd build
cmake .. -DGPU_BACKEND=CUDA -DCMAKE_INSTALL_PREFIX=<install_dir>

# Or with HIP/ROCm backend (requires ROCM_PATH env var)
cmake .. -DGPU_BACKEND=HIP -DCMAKE_INSTALL_PREFIX=<install_dir>

# Optional flags
cmake .. -DGPU_BACKEND=CUDA \
         -DENABLE_BENCHMARK=ON \
         -DENABLE_DEBUG=ON \
         -DCMAKE_INSTALL_PREFIX=<install_dir>

# Build and install
make -j$(nproc) install
```

CUDA target architectures: Volta (70), Ampere (80), Ada (89), Hopper (90).

## Usage

### C++ Interface

```cpp
#include "DedispPlan.hpp"   // or TDDPlan.hpp, FDDGPUPlan.hpp, FDDCPUPlan.hpp

// Create a plan (swap class to switch algorithm)
dedisp::DedispPlan plan(nchans, dt, f0, df, device_idx);

// Generate DM trials
plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tol);

// Execute dedispersion
plan.execute(nsamps, input, in_nbits, output, out_nbits, DEDISP_USE_DEFAULT);
```

Switching between algorithms only requires changing the plan class:

```cpp
dedisp::TDDPlan plan(nchans, dt, f0, df, device_idx);      // TDD
dedisp::FDDGPUPlan plan(nchans, dt, f0, df, device_idx);   // FDD GPU
dedisp::FDDCPUPlan plan(nchans, dt, f0, df);               // FDD CPU
```

### C Interface

A C wrapper is provided for backward compatibility:

```c
#include <dedisp.h>

dedisp_plan plan;
dedisp_create_plan(&plan, nchans, dt, f0, df);
dedisp_generate_dm_list(plan, dm_start, dm_end, pulse_width, dm_tol);
dedisp_execute(plan, nsamps, input, in_nbits, output, out_nbits, DEDISP_USE_DEFAULT);
dedisp_destroy_plan(plan);
```

To select an alternative implementation via the C API:
```c
dedisp_select_implementation(DEDISP_TDD);  // or DEDISP_FDD
// then create plan as usual
```

### FDD Runtime Options

Environment variables for the FDD implementation:

| Variable | Effect |
|---|---|
| `USE_CPU=1` | Run FDD on CPU instead of GPU |
| `USE_REFERENCE=1` | Use the reference (non-optimized) CPU implementation |
| `USE_SEGMENTED=1` | Enable time segmentation for smaller FFTs |

## Testing

```bash
cd build

# Run all registered tests
ctest

# Or run individual test binaries
<install_dir>/bin/testdedisp       # original algorithm
<install_dir>/bin/testtdd          # TDD
<install_dir>/bin/testfdd          # FDD (GPU + CPU)
<install_dir>/bin/barycenter_test  # barycentric resampling
```

## Benchmarking

Build with `-DENABLE_BENCHMARK=ON`, then run:

```bash
<install_dir>/bin/benchdedisp
<install_dir>/bin/benchtdd
<install_dir>/bin/benchfdd
```

See [bin/benchmark/README.md](bin/benchmark/README.md) for details on configuring benchmark parameters.

## Project Structure

```
src/
├── Plan.hpp / GPUPlan.hpp       Core abstract and GPU base classes
├── dedisp/                      Original dedispersion (DedispPlan)
├── tdd/                         Time Domain Dedispersion (TDDPlan)
├── fdd/                         Fourier Domain Dedispersion (FDD GPU + CPU)
├── dataHandlers/                Data I/O: abstract base, FITS, filterbank
├── presto/                      Barycentric resampling (from PRESTO)
├── numa/                        NUMA-aware memory via hwloc
├── common/                      Types, error handling, CUDA wrappers
├── gpu_macros/                  CUDA/HIP abstraction layer
└── clib/                        C API wrapper
bin/
├── test/                        Functional test applications
├── benchmark/                   Performance benchmarks
└── fil/                         Filterbank file processing
```

## Multi-GPU and Distributed Computing

- Channels are distributed across GPUs, with dedispersed time series accumulated via NCCL (CUDA) or RCCL (HIP)
- MPI integration for multi-node distributed data loading
- NUMA-aware memory placement via hwloc for optimal data locality
- Separate GPU streams for H2D transfer, compute, and D2H transfer enable pipeline overlap

## References

- Barsdell, B.R., Bailes, M., Barnes, D.G. & Fluke, C.J., 2012, "Accelerating incoherent dedispersion", [MNRAS, 422, 379](http://adsabs.harvard.edu/abs/2012arXiv1201.5380B)
- Original implementations (TDD, FDD) by ASTRON (Netherlands Institute for Radio Astronomy), 2020-2021

## License

GPL-3.0-or-later (see source headers for details).
