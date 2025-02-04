/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* CU, a CUDA driver api C++ wrapper.
* This code is copied from the IDG repository (https://git.astron.nl/RD/idg)
* and changed to meet the needs for this library.
*/
#ifndef CU_WRAPPER_H
#define CU_WRAPPER_H

#include <stdexcept>

#include <hip/hip_runtime.h>
#include <roctracer/roctracer.h>
#include <roctracer/roctracer_ext.h>
#include <roctracer/roctracer_hsa.h>

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

    void checkError();
    void checkError(hipError_t error);


    class Device {
        public:
            Device(int device);

            unsigned int get_capability();
            size_t get_total_const_memory() const;
            size_t get_free_memory() const;
            size_t get_total_memory() const;

        private:
            int m_device;
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
            HostMemory(size_t size = 0, int flags = hipHostMallocPortable);
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
            void zero(hipStream_t stream = NULL);

            template <typename T> operator T () {
                return static_cast<T>(m_ptr);
            }

        private:
            void release();
    };

    class Event {
        public:
            Event(int flags = hipEventDefault);
            ~Event();

            void synchronize();
            float elapsedTime(Event &second);

            operator hipEvent_t();

        private:
            hipEvent_t m_event;
    };

    class Stream {
        public:
            Stream(int flags = hipStreamDefault);
            ~Stream();

            void memcpyHtoDAsync(void *devPtr, const void *hostPtr, size_t size);
            void memcpyDtoHAsync(void *hostPtr, void *devPtr, size_t size);
            void memcpyDtoDAsync(void *dstPtr, void *srcPtr, size_t size);
            void memcpyHtoD2DAsync(
                void *dstPtr, size_t dstWidth,
                const void *srcPtr, size_t srcWidth,
                size_t widthBytes, size_t height);
            void memcpyDtoH2DAsync(
                void *dstPtr, size_t dstWidth,
                const void *srcPtr, size_t srcWidth,
                size_t widthBytes, size_t height);
            void memcpyHtoH2DAsync(
                void *dstPtr, size_t dstWidth,
                const void *srcPtr, size_t srcWidth,
                size_t widthBytes, size_t height);
            void memcpyDtoD2DAsync(
                void *dstPtr, size_t dstWidth,
                const void *srcPtr, size_t srcWidth,
                size_t widthBytes, size_t height);
            void synchronize();
            void waitEvent(Event &event);
            void record(Event &event);
            void zero(void *ptr, size_t size);

            operator hipStream_t();

        private:
            hipStream_t m_stream;
    };

    class Marker {
        public:
            enum Color {
              red , green, blue, yellow, black
            };

            Marker(
              const char *message,
              Marker::Color color = Color::red);

            void start();
            void end();
            void start(
              cu::Event& event);
            void end(
              cu::Event& event);

        private:
          unsigned int convert(Color color);

        protected:
          roctracer_domain_t _attributes;
          uint32_t _id;
    };

    class ScopedMarker : public Marker {
        public:

            ScopedMarker(
                const char *message,
                Marker::Color color = Color::red);

            ~ScopedMarker();

            void start() = delete;
            void end() = delete;
            void start(
              cu::Event& event) = delete;
            void end(
              cu::Event& event) = delete;
    };

} // end namespace cu

#endif // end CU_WRAPPER_H