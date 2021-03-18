/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* 2D Memory Copy helper function.
*/
#include "helper.h"
#include <unistd.h> // get total memory
#include <sys/resource.h> // get used memory

namespace dedisp
{

void memcpy2D(
    void *dstPtr, size_t dstWidth,
    const void *srcPtr, size_t srcWidth,
    size_t widthBytes, size_t height)
{
    typedef char SrcType[height][srcWidth];
    typedef char DstType[height][dstWidth];

    auto src = (SrcType *) srcPtr;
    auto dst = (DstType *) dstPtr;

    #pragma omp parallel for
    for (size_t y = 0; y < height; y++)
    {
        for (size_t x = 0; x < widthBytes; x++)
        {
            (*dst)[y][x] = (*src)[y][x];
        }
    }
}

size_t get_total_memory() {
  auto pages = sysconf(_SC_PHYS_PAGES);
  auto page_size = sysconf(_SC_PAGE_SIZE); // bits
  return pages * page_size / (1024 * 1024);  // Mbits
}

size_t get_used_memory() {
  struct rusage r_usage;
  getrusage(RUSAGE_SELF, &r_usage);  // kbits
  return r_usage.ru_maxrss / 1024;   // Mbits
}

size_t get_free_memory() { return get_total_memory() - get_used_memory(); }

} // end namespace dedisp