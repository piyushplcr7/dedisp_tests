// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
// Helper functions for FDD specifically
#ifndef FDD_HELPER_H_INCLUDE_GUARD
#define FDD_HELPER_H_INCLUDE_GUARD

#include "common/helper.h"
#include "gpu_runtime.hpp"

#if defined(USE_CUDA) || defined(USE_HIP)
struct aa_gpu_timer {
  gpuEvent_t start;
  gpuEvent_t stop;

  aa_gpu_timer() {
    gpuEventCreate(&start);
    gpuEventCreate(&stop);
  }

  ~aa_gpu_timer() {
    gpuEventDestroy(start);
    gpuEventDestroy(stop);
  }

  void Start() { gpuEventRecord(start, 0); }

  void Stop() { gpuEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed = 0.0;
    gpuEventSynchronize(stop);
    gpuEventElapsedTime(&elapsed, start, stop);
    return elapsed / 1000.0f;
  }
};
#endif // USE_CUDA || USE_HIP

#ifdef USE_OPENMP
#include <chrono>
struct aa_gpu_timer {
    std::chrono::high_resolution_clock::time_point t0, t1;
    void Start() { t0 = std::chrono::high_resolution_clock::now(); }
    void Stop()  { t1 = std::chrono::high_resolution_clock::now(); }
    float Elapsed() { return std::chrono::duration<float>(t1 - t0).count(); }
};
#endif // USE_OPENMP

namespace dedisp
{

// round up int a to a multiple of int b
inline int round_up(int a, int b)
{
    return ((a + b - 1) / b) * b;
}

template<typename InputType, typename OutputType>
void transpose_data(
    size_t height,
    size_t width,
    size_t in_stride,
    size_t out_stride,
    float offset, // use this to undo quantization, e.g. 128 for 8-bit quantization
    float scale,  // use this to prevent overflows when summing the data
    InputType *in,
    OutputType *out)
{
    #pragma omp parallel for
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            auto *src = &in[x * in_stride];
            auto *dst = &out[y * out_stride];
            dst[x] = ((OutputType) src[y] - offset) / scale;
        }
    }
}

template<typename InputType, typename OutputType>
void copy_data(
    size_t height,
    size_t width,
    size_t in_stride,
    size_t out_stride,
    InputType *in,
    OutputType *out)
{
    #pragma omp parallel for
    for (unsigned int y = 0; y < height; y++)
    {
        for (unsigned int x = 0; x < width; x++)
        {
            auto *src = &in[y * in_stride];
            auto *dst = &out[y * out_stride];
            dst[x] = src[x];
        }
    }
}

} // end namespace dedisp

#endif // FDD_HELPER_H_INCLUDE_GUARD