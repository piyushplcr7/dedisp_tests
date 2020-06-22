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
        CUresult result,
        char const *const func,
        const char *const file,
        int const line)
    {
        if (result != CUDA_SUCCESS) {
            const char *msg;
            cuGetErrorString(result, &msg);
            std::cerr << "CUDA Error at " << file;
            std::cerr << ":" << line;
            std::cerr << " in function " << func;
            std::cerr << ": " << msg;
            std::cerr << std::endl;
            throw Error<CUresult>(result);
        }
    }

    inline void __checkCudaCall(
        CUresult result,
        char const *const func,
        const char *const file,
        int const line)
    {
        try {
            __assertCudaCall(result, func, file, line);
        } catch (Error<CUresult>& error) {
            // pass
        }
    }


    /*
        HostMemory
    */
    HostMemory::HostMemory(size_t size, int flags) {
        m_capacity = size;
        m_size = size;
        m_flags = flags;
        assertCudaCall(cuMemHostAlloc(&m_ptr, size, m_flags));
    }

    HostMemory::~HostMemory() {
        release();
    }

    void HostMemory::resize(size_t size) {
        assert(size > 0);
        m_size = size;
        if (size > m_capacity) {
            release();
            assertCudaCall(cuMemHostAlloc(&m_ptr, size, m_flags));
            m_capacity = size;
        }
    }

    void HostMemory::release() {
        assertCudaCall(cuMemFreeHost(m_ptr));
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
            assertCudaCall(cuMemAlloc((CUdeviceptr *) &m_ptr, size));
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
            assertCudaCall(cuMemAlloc((CUdeviceptr *) &m_ptr, size));
            m_capacity = size;
        }
    }

    void DeviceMemory::release() {
        if (m_capacity) {
            assertCudaCall(cuMemFree((CUdeviceptr) m_ptr));
        }
    }

    void DeviceMemory::zero(CUstream stream) {
        if (m_size)
        {
            if (stream != NULL) {
                cuMemsetD8Async((CUdeviceptr) m_ptr, 0, m_size, stream);
            } else {
                cuMemsetD8((CUdeviceptr) m_ptr, 0, m_size);
            }
        }
    }


    /*
        Event
    */
    Event::Event(int flags) {
        assertCudaCall(cuEventCreate(&m_event, flags));
    }

    Event::~Event() {
        assertCudaCall(cuEventDestroy(m_event));
    }

    void Event::synchronize() {
        assertCudaCall(cuEventSynchronize(m_event));
    }

    float Event::elapsedTime(Event &second) {
        float ms;
        assertCudaCall(cuEventElapsedTime(&ms, second, m_event));
        return ms;
    }

    Event::operator CUevent() {
        return m_event;
    }


    /*
        Stream
    */
    Stream::Stream(int flags) {
        assertCudaCall(cuStreamCreate(&m_stream, flags));
    }

    Stream::~Stream() {
        assertCudaCall(cuStreamDestroy(m_stream));
    }

    void Stream::memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr, size_t size) {
        assertCudaCall(cuMemcpyHtoDAsync(devPtr, hostPtr, size, m_stream));
    }

    void Stream::memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size) {
        assertCudaCall(cuMemcpyDtoHAsync(hostPtr, devPtr, size, m_stream));
    }

    void Stream::memcpyDtoDAsync(CUdeviceptr dstPtr, CUdeviceptr srcPtr, size_t size) {
        assertCudaCall(cuMemcpyDtoDAsync(dstPtr, srcPtr, size, m_stream));
    }

    void Stream::synchronize() {
        assertCudaCall(cuStreamSynchronize(m_stream));
    }

    void Stream::waitEvent(Event &event) {
        assertCudaCall(cuStreamWaitEvent(m_stream, event, 0));
    }

    void Stream::record(Event &event) {
        assertCudaCall(cuEventRecord(event, m_stream));
    }

    Stream::operator CUstream() {
        return m_stream;
    }

} // end namespace cu
