/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* 2D Memory Copy helper function.
*/
#ifndef HELPER_H_INCLUDE_GUARD
#define HELPER_H_INCLUDE_GUARD

#include <cstddef>

#include "omp.h"

namespace dedisp
{

void memcpy2D(
    void *dstPtr, size_t dstWidth,
    const void *srcPtr, size_t srcWidth,
    size_t widthBytes, size_t height);

// Return total system (host) memory in Mbits
size_t get_total_memory();
// Return used system (host) memory in Mbits
size_t get_used_memory();
// Return free system (host) memory in Mbits
size_t get_free_memory();

} // end namespace dedisp

#endif // HELPER_H_INCLUDE_GUARD