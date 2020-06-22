#ifndef CU_WRAPPER_H
#define CU_WRAPPER_H

#include <stdexcept>

#include <cuda.h>

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
            void* ptr() { return m_ptr; }
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
            HostMemory(size_t size = 0, int flags = CU_MEMHOSTALLOC_PORTABLE);
            virtual ~HostMemory();

            void resize(size_t size) override;
            void zero();

        private:
            void release();
            int m_flags;
    };

    class DeviceMemory : public virtual Memory {

        public:
            DeviceMemory(size_t size);
            ~DeviceMemory();

            size_t capacity();
            size_t size();
            void resize(size_t size);
            void zero(CUstream stream = NULL);

        private:
            void release();
    };

    class Event {
        public:
            Event(int flags = CU_EVENT_DEFAULT);
            ~Event();

            void synchronize();
            float elapsedTime(Event &second);

            operator CUevent();

        private:
            CUevent m_event;
    };

    class Stream {
        public:
            Stream(int flags = CU_STREAM_DEFAULT);
            ~Stream();

            void memcpyHtoDAsync(CUdeviceptr devPtr, const void *hostPtr, size_t size);
            void memcpyDtoHAsync(void *hostPtr, CUdeviceptr devPtr, size_t size);
            void memcpyDtoDAsync(CUdeviceptr dstPtr, CUdeviceptr srcPtr, size_t size);
            void synchronize();
            void waitEvent(Event &event);
            void record(Event &event);

            operator CUstream();

        private:
            CUstream m_stream;
    };

} // end namespace cu

#endif // end CU_WRAPPER_H