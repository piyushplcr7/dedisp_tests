#include "CU.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cassert>

#include <vector_types.h>

#define assertCudaCall(val) __assertCudaCall(val, #val, __FILE__, __LINE__)
#define checkCudaCall(val)  __checkCudaCall(val, #val, __FILE__, __LINE__)

namespace cu {

    /*
        Error checking
    */
    inline void __assertCudaCall(
        cudaError_t result,
        char const *const func,
        const char *const file,
        int const line)
    {
        if (result != cudaSuccess) {
            const char *msg;
            msg = cudaGetErrorString(result);
            std::cerr << "CUDA Error at " << file;
            std::cerr << ":" << line;
            std::cerr << " in function " << func;
            std::cerr << ": " << msg;
            std::cerr << std::endl;
            throw Error<cudaError_t>(result);
        }
    }

    inline void __checkCudaCall(
        cudaError_t result,
        char const *const func,
        const char *const file,
        int const line)
    {
        try {
            __assertCudaCall(result, func, file, line);
        } catch (Error<cudaError_t>& error) {
            // pass
        }
    }


    /*
        Device
    */
    Device::Device(int device) {
        checkCudaCall(cudaSetDevice(device));
    }


    /*
        HostMemory
    */
    HostMemory::HostMemory(size_t size, int flags) {
        m_capacity = size;
        m_size = size;
        m_flags = flags;
        assertCudaCall(cudaHostAlloc(&m_ptr, size, m_flags));
    }

    HostMemory::~HostMemory() {
        release();
    }

    void HostMemory::resize(size_t size) {
        assert(size > 0);
        m_size = size;
        if (size > m_capacity) {
            release();
            assertCudaCall(cudaHostAlloc(&m_ptr, size, m_flags));
            m_capacity = size;
        }
    }

    void HostMemory::release() {
        assertCudaCall(cudaFreeHost(m_ptr));
    }

    void HostMemory::zero() {
        memset(m_ptr, 0, m_size);
    }


    /*
        DeviceMemory
    */
    DeviceMemory::DeviceMemory(size_t size) {
        m_capacity = size;
        m_size = size;
        if (size) {
            assertCudaCall(cudaMalloc(&m_ptr, size));
        }
    }

    DeviceMemory::~DeviceMemory() {
        release();
    }

    void DeviceMemory::resize(size_t size) {
        assert(size > 0);
        m_size = size;
        if (size > m_capacity) {
            release();
            assertCudaCall(cudaMalloc(&m_ptr, size));
            m_capacity = size;
        }
    }

    void DeviceMemory::release() {
        if (m_capacity) {
            assertCudaCall(cudaFree(m_ptr));
        }
    }

    void DeviceMemory::zero(cudaStream_t stream) {
        if (m_size)
        {
            if (stream != NULL) {
                cudaMemsetAsync(m_ptr, 0, m_size, stream);
            } else {
                cudaMemset(m_ptr, 0, m_size);
            }
        }
    }


    /*
        Event
    */
    Event::Event(int flags) {
        assertCudaCall(cudaEventCreate(&m_event, flags));
    }

    Event::~Event() {
        assertCudaCall(cudaEventDestroy(m_event));
    }

    void Event::synchronize() {
        assertCudaCall(cudaEventSynchronize(m_event));
    }

    float Event::elapsedTime(Event &second) {
        float ms;
        assertCudaCall(cudaEventElapsedTime(&ms, second, m_event));
        return ms;
    }

    Event::operator cudaEvent_t() {
        return m_event;
    }


    /*
        Stream
    */
    Stream::Stream(int flags) {
        assertCudaCall(cudaStreamCreateWithFlags(&m_stream, flags));
    }

    Stream::~Stream() {
        assertCudaCall(cudaStreamDestroy(m_stream));
    }

    void Stream::memcpyHtoDAsync(void *devPtr, const void *hostPtr, size_t size) {
        assertCudaCall(cudaMemcpyAsync(devPtr, hostPtr, size, cudaMemcpyHostToDevice, m_stream));
    }

    void Stream::memcpyDtoHAsync(void *hostPtr, void *devPtr, size_t size) {
        assertCudaCall(cudaMemcpyAsync(hostPtr, devPtr, size, cudaMemcpyDeviceToHost, m_stream));
    }

    void Stream::memcpyDtoDAsync(void *dstPtr, void *srcPtr, size_t size) {
        assertCudaCall(cudaMemcpyAsync(dstPtr, srcPtr, size, cudaMemcpyDeviceToDevice, m_stream));
    }

    void Stream::memcpyHtoD2DAsync(
        void *dstPtr, size_t dstStride,
        const void *srcPtr, size_t srcStride,
        size_t widthBytes, size_t height)
    {
        assertCudaCall(cudaMemcpy2DAsync(
            dstPtr, dstStride,
            srcPtr, srcStride,
            widthBytes, height,
            cudaMemcpyHostToDevice));
    }

    void Stream::memcpyDtoH2DAsync(
        void *dstPtr, size_t dstStride,
        const void *srcPtr, size_t srcStride,
        size_t widthBytes, size_t height)
    {
        assertCudaCall(cudaMemcpy2DAsync(
            dstPtr, dstStride,
            srcPtr, srcStride,
            widthBytes, height,
            cudaMemcpyDeviceToHost));
    }

    void Stream::synchronize() {
        assertCudaCall(cudaStreamSynchronize(m_stream));
    }

    void Stream::waitEvent(Event &event) {
        assertCudaCall(cudaStreamWaitEvent(m_stream, event, 0));
    }

    void Stream::record(Event &event) {
        assertCudaCall(cudaEventRecord(event, m_stream));
    }

    Stream::operator cudaStream_t() {
        return m_stream;
    }

} // end namespace cu
