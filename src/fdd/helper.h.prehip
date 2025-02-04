// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
// Helper functions for FDD specifically
#ifndef FDD_HELPER_H_INCLUDE_GUARD
#define FDD_HELPER_H_INCLUDE_GUARD

#include "common/helper.h"
#include <cuda_runtime.h>

struct aa_gpu_timer {
  cudaEvent_t start;
  cudaEvent_t stop;

  aa_gpu_timer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~aa_gpu_timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed = 0.0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed / 1000.0f;
  }
};

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