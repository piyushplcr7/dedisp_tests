#ifndef CU_WRAPPER_H
#define CU_WRAPPER_H

#include <stdexcept>

#include <cuda_runtime.h>

namespace cu {

    template<typename T>
    class Error : public std::exception {
        public:
            Error(T result):
            _result(result) {}

            operator T() const {
                return _result;
            }

        private:
            T _result;
    };

    class Memory {
        public:
            void* data() { return m_ptr; }
            size_t size() { return m_size; }
            virtual void resize(size_t size) = 0;
            template <typename T> operator T *() {
                return static_cast<T *>(m_ptr);
            }

        protected:
            size_t m_capacity = 0; // capacity, in bytes
            size_t m_size = 0; // size, in bytes
            void*  m_ptr = nullptr;
    };

    class HostMemory : public virtual Memory {
        public:
            HostMemory(size_t size = 0, int flags = cudaHostAllocDefault);
            virtual ~HostMemory();

            void resize(size_t size) override;
            void zero();

        private:
            void release();
            int m_flags;
    };

    class DeviceMemory : public virtual Memory {

        public:
            DeviceMemory(size_t size = 0);
            ~DeviceMemory();

            void resize(size_t size);
            void zero(cudaStream_t stream = NULL);

            template <typename T> operator T () {
                return static_cast<T>(m_ptr);
            }

        private:
            void release();
    };

    class Event {
        public:
            Event(int flags = cudaEventDefault);
            ~Event();

            void synchronize();
            float elapsedTime(Event &second);

            operator cudaEvent_t();

        private:
            cudaEvent_t m_event;
    };

    class Stream {
        public:
            Stream(int flags = cudaStreamDefault);
            ~Stream();

            void memcpyHtoDAsync(void *devPtr, const void *hostPtr, size_t size);
            void memcpyDtoHAsync(void *hostPtr, void *devPtr, size_t size);
            void memcpyDtoDAsync(void *dstPtr, void *srcPtr, size_t size);
            void memcpyHtoD2DAsync(
                void *dstPtr, size_t dstStride,
                const void *srcPtr, size_t srcStride,
                size_t width_bytes, size_t height);
            void memcpyDtoH2DAsync(
                void *dstPtr, size_t dstStride,
                const void *srcPtr, size_t srcStride,
                size_t width_bytes, size_t height);
            void synchronize();
            void waitEvent(Event &event);
            void record(Event &event);

            operator cudaStream_t();

        private:
            cudaStream_t m_stream;
    };

} // end namespace cu

#endif // end CU_WRAPPER_H