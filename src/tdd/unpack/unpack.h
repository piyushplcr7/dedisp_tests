/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* Time Domain Dedispersion (TDD)
* is an optimized version of the original dedisp implementation.
*/
#include "dedisp_types.h"
#include "cuda_runtime.h"

void transpose_unpack(
    const dedisp_word* d_in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    dedisp_word* d_out,
    dedisp_size in_nbits, dedisp_size out_nbits,
    cudaStream_t stream);
