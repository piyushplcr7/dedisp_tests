#include <iostream>
#include <hwloc.h>
#include "gpu_runtime.hpp"
#include <numa.h>
#include <vector>
#include "numa/hwloc_utils.hpp"
#include "numa/numa_mem.hpp"
#include <thread>


hwlocUtils hwlocobj{};

void testMemCpy(int node, int cuda_dev, bool concurrent) {

    if (node < 0)
        return;
    
    hwlocobj.bindThreadToNUMA(node);
    hwlocobj.bindMemToNUMA(node);

    gpuSetDevice(cuda_dev);

    size_t gigabytes = 8;
    const size_t bytes = gigabytes * 1024 * 1024 * 1024;

    numaMem h2dmem(node, bytes);
    numaMem d2hmem(node, bytes);

    void* hptr_h2d = h2dmem.get();
    void* hptr_d2h = d2hmem.get();

    h2dmem.touchPages();
    d2hmem.touchPages();

    gpuHostRegister(hptr_h2d, bytes, gpuHostRegisterPortable);
    gpuHostRegister(hptr_d2h, bytes, gpuHostRegisterPortable);

    void* dptr_h2d;
    void* dptr_d2h;
    gpuMalloc(&dptr_h2d, bytes);
    gpuMalloc(&dptr_d2h, bytes);
    
    int trials = 10;

    std::vector<gpuEvent_t> starth2d(trials), stoph2d(trials);
    std::vector<gpuEvent_t> startd2h(trials), stopd2h(trials);

    for (int i = 0 ; i < trials ; ++i) {
        gpuEventCreate(&starth2d[i]);
        gpuEventCreate(&stoph2d[i]);
        
        gpuEventCreate(&startd2h[i]);
        gpuEventCreate(&stopd2h[i]);
    }

    gpuStream_t streamH2D, streamD2H;
    gpuStreamCreate(&streamH2D);

    if (concurrent) {
        gpuStreamCreate(&streamD2H);
    }
    else {
        streamD2H = streamH2D;
    }

    // Warmup
    gpuMemset(dptr_h2d, 0, bytes);
    gpuMemset(dptr_d2h, 0, bytes);
    gpuDeviceSynchronize();
    gpuMemcpyAsync(dptr_h2d, hptr_h2d, bytes, gpuMemcpyHostToDevice, streamH2D);
    gpuMemcpyAsync(hptr_d2h, dptr_d2h, bytes, gpuMemcpyDeviceToHost, streamD2H);
    gpuDeviceSynchronize();

    for (int i = 0 ; i < trials ; ++i) {
        gpuEventRecord(starth2d[i], streamH2D);
        gpuMemcpyAsync(dptr_h2d, hptr_h2d, bytes, gpuMemcpyHostToDevice, streamH2D);
        gpuEventRecord(stoph2d[i], streamH2D);

        gpuEventRecord(startd2h[i], streamD2H);
        gpuMemcpyAsync(hptr_d2h, dptr_d2h, bytes, gpuMemcpyDeviceToHost, streamD2H);
        gpuEventRecord(stopd2h[i], streamD2H);
    }

    gpuStreamSynchronize(streamH2D);
    gpuStreamSynchronize(streamD2H);

    float totmsh2d=0., totmsd2h=0.;
    float ms;
    for (int i = 0 ; i < trials ; ++i) {
        gpuEventElapsedTime(&ms, starth2d[i], stoph2d[i]);
        totmsh2d += ms;

        gpuEventElapsedTime(&ms, startd2h[i], stopd2h[i]);
        totmsd2h += ms;
    }
    
    totmsh2d /= trials;
    totmsd2h /= trials;

    std::cout << "Numa node " << node << " <-> cuda device " << cuda_dev << ", concurrent = " << concurrent << std::endl;
    std::cout << "H2D: " << gigabytes / (totmsh2d/1e3) << " GBps" << std::endl;
    std::cout << "D2H: " << gigabytes / (totmsd2h/1e3) << " GBps" << std::endl;

    for (int i = 0 ; i < trials ; ++i) {
        gpuEventDestroy(starth2d[i]);
        gpuEventDestroy(stoph2d[i]);
        gpuEventDestroy(startd2h[i]);
        gpuEventDestroy(stopd2h[i]);
    }
    gpuStreamDestroy(streamH2D);
    if (concurrent)
        gpuStreamDestroy(streamD2H);

    gpuFree(dptr_h2d);
    gpuFree(dptr_d2h);

    gpuHostUnregister(hptr_h2d);
    gpuHostUnregister(hptr_d2h);
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
