# Introduction #

dedisp provides a simple C and C++ interface to computing dedispersion transforms using a GPU. The interface is modelled on that of the well-known [FFTW](http://www.fftw.org/) library, and uses an object-oriented approach. The user first creates a dedispersion _plan_, and then calls functions to modify or execute that plan. A full list and description of the functions provided by the library can be viewed in the API documentation below. Also refer to the sources of the test applications for an example on how to use the library. *This fork* adds an improved (performance) implementation of dedisp, referred to as Time Domain Dedispersion (TDD) and adds a new dedispersion algorithm for Fourier Domain Dedispersion (FDD). By default a dedispersion plan will execute the original dedispersion implementation. The alternative implementations can be selected at will. Refer to the description of alternative implementations below for more information.

# Dependencies #

The library requires NVIDIA's [CUDA](https://developer.nvidia.com/cuda-zone) in order to access the GPU. This also imposes the constraint that the target hardware must be an NVIDIA GPU. To compile the library you will have to configure the CUDA paths in the CMAKE configuration dialogue.

# Installation #

Refer to the main [README](README.md) for installation instructions.
If all goes well, the library will be built in the lib/ subdir and the header in include/ subdir. If something goes wrong, please let me know either via email or the project website.

# Testing #

This repository contains several different test applications:
* functional test applicatiosn with generated data [README](bin/test/README.md)
* functional test applications with a filterbank file as input [README](bin/fil/README.md)
* performance test applications with dummy input [README](bin/benchmark/README.md)

# Debugging #

The following cmake flag can be used to build the library in a verbose debug mode: set `DEDISP_DEBUG` to `ON`.

# API Documentation #

Both the C and C++ interfaces are documented in their headerfiles. The C-API also has extensive doxygen documentation which can be generated locally.

# C++ Interface #

The original repository was modified from C to C++ to allow more flexibility in adding alternative implementations. The default interface of the library is therefore based on C++. There is one main interface `dedisp::Plan` to the library. Implementation specific interfaces inherit from this class. E.g. to use the original dedispersion implementation one creates a `dedisp::DedispPlan plan(nchans, dt, f0, df, device_idx)` and to use the optimized TDD implementation one creates a `dedisp::TDDPlan plan(nchans, dt, f0, df, device_idx)`. This makes that one can easilly switch between different implementations for dedispersion withouth changing the interface (as is demonstrated with the generic test application in test.hpp).
Refer to the test applications for examples of usage and refer to the generic sourcefile `Plan.hpp` and implementation specific sourcefiles `DedispPlan.hpp`, `TDDPlan.hpp`, `FDDCPUPlan.hpp` and `FDDGPUPlan.hpp` for a more elaborate description of the interfaces.

# C Interface #

For backwards compatibility we have included a C wrapper around the C++ interfaces. Both the C and the C++ symbols are included in libdedisp.so, meaning that your original application should not be changed to work with the new library (when using the dedisp implemenation as default). Refer to `ctestdedisp` for an example application with the C interface and refer to `dedisp.h` for the full interface description. The C wrapper interface creates a `dedisp::DedispPlan` by default. Call `dedisp_select_implementation(<implementation>)` with `implementation = DEDISP_FDD or DEDISP_TDD` *before* plan creation to select an alternative dedispersion implementation.

# Alternative Implementations #

By ASTRON (Netherlands Institute for Radio Astronomy) 2020-2021, publication pending.

We have created an optimized version of the dedisp library next to the original implementation, we refer to this as TDD (Time-Domain Dedispersion) and to the original implementation as dedisp. Next to the TDD implementation we have also implemented our own dedispersion algorithm where dedisperion is performed in the Fourier-domain, we refer to this implementation as FDD.

We have tested the three implementations on NVIDIA Titan RTX with CUDA 11. Our TDD implementation is a factor 1.5x to 2x faster than dedisp (depending on `#DMs`). FDD is again up to 20% faster than TDD, for `#DMs > 512`, and is expected to scale better with advancements in GPU. Where dedisp and TDD are memory bound, FDD is compute bound.

## TDD #

We made the following changes to dedisp:
1) transferring of input and output data were overlapped with compute on the GPU, moving input and output copy out of the critical path;
2) memory transfers were changed from paged to pinned memory, thus increasing transfer speeds by up to a factor of 2 to 3;
3) the unpack and transpose operations were combined in to one kernel thus requiring only a single pass over the data;
4) the usage of texture memory in the dedispersion kernel was made optional and disabled for our implementation (Pascal architecture benefits from texture memory, Turing is faster when it is disabled).

TDD is functionally the same as the original dedisp and produces the same result with both libdedisp testcases and original .fil data.

TDD can be used through the C++ interface with `dedisp::TDDPlan plan(nchans, dt, f0, df, device_idx)` or through the C interface with `dedisp_select_implementation(DEDISP_TDD)`.

## FDD #

We have tested our approach with Fourier-domain dedispersion. In FDD, the shifts of time series at individual observing frequencies are replaced by phase rotations, implemented as multiplication by phasors, by Fourier transforming each time series from the time-domain to the spin frequency-domain. This requires much less I/O and enables more efficient computations. The final results are transferred back to the time-domain, or can be kept in the Fourier-domain for further processing. We will describe the algorithm and implementation in more detail in our publication.

FDD is functionally similar to the original dedisp. The testcases of libdedisp show differences with the original, but the same DMs are found in both generated test data and original filterbank data. Also on further analysis of the output we find the same results with FDD as with the original implementation. If you are interested in this implementation please test it with your pipeline. You can contact us for more information.

FDD can be used through the C++ interface with `dedisp::FDDGPUPlan plan(nchans, dt, f0, df, device_idx)` or through the C interface with `dedisp_select_implementation(DEDISP_FDD)`.

The implementation of FDD also contains several reference and experimental implementations. The FDD implementations are implemented for CPU as well to allow easier testing or to perform dedispersion on a system withouth a GPU. The CPU implementation uses OpenMP for parallelization and the FFTW library for the FFTs. The GPU implementation uses the cuFFT library for the FFTs.

### CPU
These are methods from `dedisp::FDDCPUPlan`
- `FDDCPUPlan dedisperse_reference()`
  - Might be enabled at runtime through setting envirionment variable: `USE_REFERENCE=1`
  - This is a straight forward reference implementation of the FDD algorithm, only simple optimizations are applied
- `FDDCPUPlan dedisperse_optimized()`
  - This is the default implementation for `FDDCPUPlan`
  - This is the DM batched optimized implementation
  - This implementation is similar to the default GPU implementation
  - Additional *extrapolation feature* might be applied by setting a templated function parameter (in source) to true (more on the extrapolation feature below)
  - This is what you get when running `testfdd` with envirionment variable: `USE_CPU=1`
- `FDDCPUPlan dedisperse_segmented_reference()`
  - Might be enabled at runtime through setting envirionment variables: `USE_REFERENCE=1` and `USE_SEGMENTED=1`
  - Alternative batching method that divides the input in segments of time samples, refer to below
  - This is a straight forward reference implementation of the segementation feature, only simple optimizations are applied
- `FDDCPUPlan dedisperse_segmented_optimized()`
  - Might be enabled at runtime through setting envirionment variable: `USE_SEGMENTED=1`
  - This is the optimized implementation for the time segmentation feature

### GPU
These are methods from `dedisp::FDDGPUPlan`
- `FDDGPUPlan execute_gpu()`
  - This is the default FDDGPUPlan implementation
  - This implementation is similar to `FDDCPUPlan dedisperse_optimized()`
  - Additional *extrapolation feature* might be applied by setting a kernel parameter to true in sourcefile `FDDKernel.cu` (more on the extrapolation feature below)
  - This is what you get when running `testfdd`
- `FDDGPUPlan execute_gpu_segmented()`
  - Might be enabled at runtime through setting envirionment variable: `USE_SEGMENTED=1`
  - This implementation is similar to `FDDCPUPlan dedisperse_segmented_optimized()`
  - Uses the same GPU kernels as the non-segmented implementation

### Time segmentation

This is an experimental optimization feature, where input samples are divided in to nicely dimensioned segments (time samples) and then processed for all DMs.
This feature allows to only copy input data to the GPU once. Contrary to the current approach where for large amounts of trial-DMs we introduce an outer DM job to overcome GPU memory size limitations. This separation in outer DM jobs requires an additional pass/passess over the input data which might lead to inefficiency. However we are able to overlap transfer and compute well, thus minimizing this inefficiency.
Also segmentation allows for smaller sized (more efficient) FFTs speeding up the pre-processing phase. Still, we observe that the current implementation (`FDDGPUPlan execute_gpu()`) with dimensioning in DM outer and inner jobs performs better for the NVIDIA Titan RTX. We are leaving this feature in because the balance might be different for other GPU Architectures.
Note the time segmentation feature might miss very large DMs when using small segments of input data.

### Extrapolation

This is an experimental optimization feature, where extrapolation is used in the computation of the phasors in the dedispersion kernel. Extrapolation of the phasors might allow to reach a better balance in sine and cosine operations vs multiply and accumulate operations at the cost of a small inaccuracy. This feature should be further explored to determine whether functional correctness is achieved at all times. With the NVIDIA Titan RTX we observe a marginal performance improvement with this feature. We are leaving this feature in because it might be beneficial depending on the system configuration.