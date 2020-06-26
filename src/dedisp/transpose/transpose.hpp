/*
 * This file exposes the generic Transpose<T>::transpose method
 *
 * Explicit template instantiations for this method are defined in transpose.cu
 */

template<typename T>
void transpose(
    const T* in,
    size_t width, size_t height,
    size_t in_stride, size_t out_stride,
    T* out,
    cudaStream_t stream=0);