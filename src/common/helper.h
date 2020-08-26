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

} // end namespace dedisp

#endif // HELPER_H_INCLUDE_GUARD