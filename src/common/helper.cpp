#include "helper.h"

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

} // end namespace dedisp