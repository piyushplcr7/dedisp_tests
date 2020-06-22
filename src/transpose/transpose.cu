/*
 * This header files contains the implementation of the Transpose<T> class
 */
#include "transpose.cuh"

/*
 * The generic Transpose<T>::transpose method
 */
template<typename T>
void transpose(
    const T* in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    T* out,
    cudaStream_t stream=0)
{
    Transpose<T> transpose;
    transpose.transpose(in, width, height, in_stride, out_stride, out, stream);
}

/*
 * Explicit template instantations
 */
template void transpose<unsigned int>(const unsigned int*, size_t, size_t, size_t, size_t, unsigned int*, cudaStream_t);