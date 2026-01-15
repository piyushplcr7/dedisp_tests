#include <iostream>
#include <hwloc.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <vector>
#include "numa/hwloc_utils.hpp"
#include "numa/numa_mem.hpp"
#include <thread>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " \
                  << cudaGetErrorString(err) << std::endl; \
        std::abort(); \
    } \
} while (0)

hwlocUtils hwlocobj{};

void testMemCpy(int node, int cuda_dev, bool concurrent) {

    if (node < 0)
        return;
    
    hwlocobj.bindThreadToNUMA(node);
    hwlocobj.bindMemToNUMA(node);

    CUDA_CHECK(cudaSetDevice(cuda_dev));

    size_t gigabytes = 8;
    const size_t bytes = gigabytes * 1024 * 1024 * 1024;

    numaMem h2dmem(node, bytes);
    numaMem d2hmem(node, bytes);

    void* hptr_h2d = h2dmem.get();
    void* hptr_d2h = d2hmem.get();

    h2dmem.touchPages();
    d2hmem.touchPages();

    CUDA_CHECK(cudaHostRegister(hptr_h2d, bytes, cudaHostRegisterPortable));
    CUDA_CHECK(cudaHostRegister(hptr_d2h, bytes, cudaHostRegisterPortable));

    void* dptr_h2d;
    void* dptr_d2h;
    CUDA_CHECK(cudaMalloc(&dptr_h2d, bytes));
    CUDA_CHECK(cudaMalloc(&dptr_d2h, bytes));
    
    int trials = 10;

    std::vector<cudaEvent_t> starth2d(trials), stoph2d(trials);
    std::vector<cudaEvent_t> startd2h(trials), stopd2h(trials);

    for (int i = 0 ; i < trials ; ++i) {
        cudaEventCreate(&starth2d[i]);
        cudaEventCreate(&stoph2d[i]);
        
        cudaEventCreate(&startd2h[i]);
        cudaEventCreate(&stopd2h[i]);
    }

    cudaStream_t streamH2D, streamD2H;
    CUDA_CHECK(cudaStreamCreate(&streamH2D));

    if (concurrent) {
        CUDA_CHECK(cudaStreamCreate(&streamD2H));
    }
    else {
        streamD2H = streamH2D;
    }

    // Warmup
    CUDA_CHECK(cudaMemset(dptr_h2d, 0, bytes));
    CUDA_CHECK(cudaMemset(dptr_d2h, 0, bytes));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpyAsync(dptr_h2d, hptr_h2d, bytes, cudaMemcpyHostToDevice, streamH2D));
    CUDA_CHECK(cudaMemcpyAsync(hptr_d2h, dptr_d2h, bytes, cudaMemcpyDeviceToHost, streamD2H));
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0 ; i < trials ; ++i) {
        CUDA_CHECK(cudaEventRecord(starth2d[i], streamH2D));
        CUDA_CHECK(cudaMemcpyAsync(dptr_h2d, hptr_h2d, bytes, cudaMemcpyHostToDevice, streamH2D));
        CUDA_CHECK(cudaEventRecord(stoph2d[i], streamH2D));

        CUDA_CHECK(cudaEventRecord(startd2h[i], streamD2H));
        CUDA_CHECK(cudaMemcpyAsync(hptr_d2h, dptr_d2h, bytes, cudaMemcpyDeviceToHost, streamD2H));
        CUDA_CHECK(cudaEventRecord(stopd2h[i], streamD2H));
    }

    CUDA_CHECK(cudaStreamSynchronize(streamH2D));
    CUDA_CHECK(cudaStreamSynchronize(streamD2H));

    float totmsh2d=0., totmsd2h=0.;
    float ms;
    for (int i = 0 ; i < trials ; ++i) {
        CUDA_CHECK(cudaEventElapsedTime(&ms, starth2d[i], stoph2d[i]));
        totmsh2d += ms;

        CUDA_CHECK(cudaEventElapsedTime(&ms, startd2h[i], stopd2h[i]));
        totmsd2h += ms;
    }
    
    totmsh2d /= trials;
    totmsd2h /= trials;

    std::cout << "Numa node " << node << " <-> cuda device " << cuda_dev << ", concurrent = " << concurrent << std::endl;
    std::cout << "H2D: " << gigabytes / (totmsh2d/1e3) << " GBps" << std::endl;
    std::cout << "D2H: " << gigabytes / (totmsd2h/1e3) << " GBps" << std::endl;

    for (int i = 0 ; i < trials ; ++i) {
        CUDA_CHECK(cudaEventDestroy(starth2d[i]));
        CUDA_CHECK(cudaEventDestroy(stoph2d[i]));
        CUDA_CHECK(cudaEventDestroy(startd2h[i]));
        CUDA_CHECK(cudaEventDestroy(stopd2h[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(streamH2D));
    if (concurrent)
        CUDA_CHECK(cudaStreamDestroy(streamD2H));

    cudaFree(dptr_h2d);
    cudaFree(dptr_d2h);

    CUDA_CHECK(cudaHostUnregister(hptr_h2d));
    CUDA_CHECK(cudaHostUnregister(hptr_d2h));
}

int main(int argc, char** argv)
{   
    if (argc != 6) {
        std::cerr << "Usage: ./executable a b c d concurrent, where a-d are numa nodes, concurrent is boolean" << std::endl;
        exit(-1); 
    }

    int a = atoi(argv[1]);
    int b = atoi(argv[2]);
    int c = atoi(argv[3]);
    int d = atoi(argv[4]);

    bool concurrent;

    if (strcmp(argv[5],"true") == 0) {
        concurrent = true;
    }
    else if (strcmp(argv[5],"false") == 0) {
        concurrent = false;
    }
    else {
        std::cerr << "invalid argument 5, it must be true or false" << std::endl;
        exit(-1);
    }

    std::vector<int> cudaDevices = {0,1,2,3};
    std::vector<int> NUMAnodes = {a, b, c, d};

    std::vector<std::thread> threads(cudaDevices.size());

    for (int i = 0 ; i < cudaDevices.size() ; ++i) {
        threads[i] = std::thread(testMemCpy, NUMAnodes[i], cudaDevices[i], concurrent);
    }

    for (int i = 0 ; i < cudaDevices.size() ; ++i) 
        threads[i].join();
    
    return 0;
}
