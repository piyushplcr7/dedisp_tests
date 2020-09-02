#include "CU.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cassert>

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

    void checkError()
    {
        assertCudaCall(cudaGetLastError());
    }

    void checkError(cudaError_t error)
    {
        assertCudaCall(error);
    }


    /*
        Device
    */
    Device::Device(int device) {
        m_device = device;
        checkCudaCall(cudaSetDevice(device));
    }

    unsigned int Device::get_capability() {
        cudaDeviceProp device_props;
        cudaGetDeviceProperties(&device_props, m_device);
        return 10 * device_props.major +
                    device_props.minor;
    }

    size_t Device::get_total_const_memory() const {
        cudaDeviceProp device_props;
        cudaGetDeviceProperties(&device_props, m_device);
        return device_props.totalConstMem;
    }

    size_t Device::get_free_memory() const {
        size_t free;
        size_t total;
        cudaMemGetInfo(&free, &total);
        return free;
    }

    size_t Device::get_total_memory() const {
        size_t free;
        size_t total;
        cudaMemGetInfo(&free, &total);
        return total;
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
                assertCudaCall(cudaMemsetAsync(m_ptr, 0, m_size, stream));
            } else {
                assertCudaCall(cudaMemset(m_ptr, 0, m_size));
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
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height)
    {
        assertCudaCall(cudaMemcpy2DAsync(
            dstPtr, dstWidth,
            srcPtr, srcWidth,
            widthBytes, height,
            cudaMemcpyHostToDevice,
            m_stream));
    }

    void Stream::memcpyDtoH2DAsync(
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height)
    {
        assertCudaCall(cudaMemcpy2DAsync(
            dstPtr, dstWidth,
            srcPtr, srcWidth,
            widthBytes, height,
            cudaMemcpyDeviceToHost,
            m_stream));
    }

    void Stream::memcpyHtoH2DAsync(
        void *dstPtr, size_t dstWidth,
        const void *srcPtr, size_t srcWidth,
        size_t widthBytes, size_t height)
    {
        assertCudaCall(cudaMemcpy2DAsync(
            dstPtr, dstWidth,
            srcPtr, srcWidth,
            widthBytes, height,
            cudaMemcpyHostToHost,
            m_stream));
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

    void Stream::zero(void *ptr, size_t size) {
        assertCudaCall(cudaMemsetAsync(ptr, 0, size, m_stream));
    }

    Stream::operator cudaStream_t() {
        return m_stream;
    }


    /*
        Marker
    */
    Marker::Marker(
      const char *message,
      Color color)
    {
      _attributes.version       = NVTX_VERSION;
      _attributes.size          = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
      _attributes.colorType     = NVTX_COLOR_ARGB;
      _attributes.color         = convert(color);
      _attributes.messageType   = NVTX_MESSAGE_TYPE_ASCII;
      _attributes.message.ascii = message;
    }

    void Marker::start()
    {
      _id = nvtxRangeStartEx(&_attributes);
    }

    void Marker::end()
    {
      nvtxRangeEnd(_id);
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
        _id = nvtxRangeStartEx(&_attributes);
      };

    ScopedMarker::~ScopedMarker()
    {
      nvtxRangeEnd(_id);
    }

} // end namespace cu
