/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* CU, a CUDA driver api C++ wrapper.
* This code is copied from the IDG repository (https://git.astron.nl/RD/idg)
* and changed to meet the needs for this library.
*/
#include "CU.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cassert>
#include "gpu_runtime.hpp"

#define assertCudaCall(val) __assertCudaCall(val, #val, __FILE__, __LINE__)
#define checkCudaCall(val)  __checkCudaCall(val, #val, __FILE__, __LINE__)

namespace cu {

    /*
        Error checking
    */
    inline void __assertCudaCall(
        gpuError_t result,
        char const *const func,
        const char *const file,
        int const line)
    {
        if (result != gpuSuccess) {
            const char *msg;
            msg = gpuGetErrorString(result);
            std::cerr << "CUDA Error at " << file;
            std::cerr << ":" << line;
            std::cerr << " in function " << func;
            std::cerr << ": " << msg;
            std::cerr << std::endl;
            throw Error<gpuError_t>(result);
        }
    }

    inline void __checkCudaCall(
        gpuError_t result,
        char const *const func,
        const char *const file,
        int const line)
    {
        try {
            __assertCudaCall(result, func, file, line);
        } catch (Error<gpuError_t>& error) {
            std::cout << error.what() << std::endl;
        }
    }

    void checkError()
    {
        gpuGetLastError();
    }

    void checkError(gpuError_t error)
    {
        assertCudaCall(error);
    }


    /*
        Device
    */
    Device::Device(int device) {
        m_device = device;
        gpuSetDevice(device);
    }

    unsigned int Device::get_capability() {
        gpuDeviceProp_t device_props;
        gpuGetDeviceProperties(&device_props, m_device);
        return 10 * device_props.major +
                    device_props.minor;
    }

    size_t Device::get_total_const_memory() const {
        gpuDeviceProp_t device_props;
        gpuGetDeviceProperties(&device_props, m_device);
        return device_props.totalConstMem;
    }

    size_t Device::get_free_memory() const {
        size_t free;
        size_t total;
        gpuMemGetInfo(&free, &total);
        return free;
    }

    size_t Device::get_total_memory() const {
        size_t free;
        size_t total;
        gpuMemGetInfo(&free, &total);
        return total;
    }


    /*
        HostMemory
    */
    HostMemory::HostMemory(size_t size, int flags) {
        m_capacity = size;
        m_size = size;
        m_flags = flags;
        gpuHostAlloc(&m_ptr, size, m_flags);
    }

    HostMemory::~HostMemory() {
        release();
    }

    void HostMemory::resize(size_t size) {
        assert(size > 0);
        m_size = size;
        if (size > m_capacity) {
            release();
            gpuHostAlloc(&m_ptr, size, m_flags);
            m_capacity = size;
        }
    }

    void HostMemory::release() {
        gpuHostFree(m_ptr);
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
            gpuMalloc(&m_ptr, size);
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
            gpuMalloc(&m_ptr, size);
            m_capacity = size;
        }
    }

    DeviceMemory::DeviceMemory(size_t size, int dev_id) {
        m_capacity = size;
        m_size = size;
        if (size) {
            gpuSetDevice(dev_id);                  // set device for this thread
            gpuMalloc(&m_ptr, size);
        }
    }

    void DeviceMemory::resize(size_t size, int dev_id) {
        assert(size > 0);
        m_size = size;
        if (size > m_capacity) {
            release();
            gpuSetDevice(dev_id);                  // ensure allocation goes to correct device
            gpuMalloc(&m_ptr, size);
            m_capacity = size;
        }
    }

    void DeviceMemory::release() {
        if (m_capacity) {
            gpuFree(m_ptr);
        }
    }

    void DeviceMemory::zero(gpuStream_t stream) {
        if (m_size)
        {
            if (stream != NULL) {
                gpuMemsetAsync(m_ptr, 0, m_size, stream);
            } else {
                gpuMemset(m_ptr, 0, m_size);
            }
        }
    }


    /*
        Event
    */
    Event::Event(int flags) {
        gpuEventCreateWithFlags(&m_event, flags);
    }

    Event::~Event() {
        gpuEventDestroy(m_event);
    }

    void Event::synchronize() {
        gpuEventSynchronize(m_event);
    }

    float Event::elapsedTime(Event &second) {
        float ms;
        gpuEventElapsedTime(&ms, second, m_event);
        return ms;
    }

    Event::operator gpuEvent_t() {
        return m_event;
    }


    /*
        Stream
    */
    Stream::Stream(int flags) {
        gpuStreamCreateWithFlags(&m_stream, flags);
    }

    Stream::~Stream() {
        gpuStreamDestroy(m_stream);
    }

    void Stream::memcpyHtoDAsync(void *devPtr, const void *hostPtr, size_t size) {
        gpuMemcpyAsync(devPtr, hostPtr, size, gpuMemcpyHostToDevice, m_stream);
    }

    void Stream::memcpyDtoHAsync(void *hostPtr, void *devPtr, size_t size) {
        gpuMemcpyAsync(hostPtr, devPtr, size, gpuMemcpyDeviceToHost, m_stream);
    }

    void Stream::memcpyDtoDAsync(void *dstPtr, void *srcPtr, size_t size) {
        gpuMemcpyAsync(dstPtr, srcPtr, size, gpuMemcpyDeviceToDevice, m_stream);
    }

    void Stream::memcpyHtoD2DAsync(
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height)
    {
        gpuMemcpy2DAsync(
            dstPtr, dstWidth,
            srcPtr, srcWidth,
            widthBytes, height,
            gpuMemcpyHostToDevice,
            m_stream);
    }

    void Stream::memcpyDtoH2DAsync(
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height)
    {
        gpuMemcpy2DAsync(
            dstPtr, dstWidth,
            srcPtr, srcWidth,
            widthBytes, height,
            gpuMemcpyDeviceToHost,
            m_stream);
    }

    void Stream::memcpyHtoH2DAsync(
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height)
    {
        gpuMemcpy2DAsync(
            dstPtr, dstWidth,
            srcPtr, srcWidth,
            widthBytes, height,
            gpuMemcpyHostToHost,
            m_stream);
    }

    void Stream::memcpyDtoD2DAsync(
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height)
    {
        gpuMemcpy2DAsync(
            dstPtr, dstWidth,
            srcPtr, srcWidth,
            widthBytes, height,
            gpuMemcpyDeviceToDevice,
            m_stream);
    }

    void Stream::synchronize() {
        gpuStreamSynchronize(m_stream);
    }

    void Stream::waitEvent(Event &event) {
        gpuStreamWaitEvent(m_stream, event, 0);
    }

    void Stream::record(Event &event) {
        gpuEventRecord(event, m_stream);
    }

    void Stream::zero(void *ptr, size_t size) {
        gpuMemsetAsync(ptr, 0, size, m_stream);
    }

    Stream::operator gpuStream_t() {
        return m_stream;
    }


    /*
        Marker
    */
    Marker::Marker(
      const char *message,
      Color color)
    {
#ifdef USE_CUDA
      _attributes.version       = NVTX_VERSION;
      _attributes.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      _attributes.colorType     = NVTX_COLOR_ARGB;
      _attributes.color         = convert(color);
      _attributes.messageType   = NVTX_MESSAGE_TYPE_ASCII;
      _attributes.message.ascii = message;
#endif
    }

    void Marker::start()
    {
#ifdef USE_CUDA
      _id = nvtxRangeStartEx(&_attributes);
#endif

#ifdef USE_HIP
      roctracer_start();
#endif
    }

    void Marker::end()
    {
#ifdef USE_CUDA
      nvtxRangeEnd(_id);
#endif

#ifdef USE_HIP
      roctracer_stop();
#endif
    }

    void Marker::start(
      cu::Event& event)
    {
      event.synchronize();
      start();
    }

    void Marker::end(
      cu::Event& event)
    {
      event.synchronize();
      end();
    }

    unsigned int Marker::convert(Color color)
    {
        switch (color) {
          case red :    return 0xffff0000;
          case green :  return 0xff00ff00;
          case blue :   return 0xff0000ff;
          case yellow : return 0xffffff00;
          case black :  return 0xff000000;
          default:      return 0xff00ff00;
        }
    }


    /*
        ScopedMarker
    */
    ScopedMarker::ScopedMarker(
      const char *message,
      Color color) :
      Marker(message, color)
      {
#ifdef USE_CUDA
        _id = nvtxRangeStartEx(&_attributes);
#endif

#ifdef USE_HIP
        roctracer_start();
#endif
      };

    ScopedMarker::~ScopedMarker()
    {
#ifdef USE_CUDA
      nvtxRangeEnd(_id);
#endif

#ifdef USE_HIP
      roctracer_stop();
#endif
    }

} // end namespace cu
