#include <iostream>
#include <hwloc.h>
#include "gpu_runtime.hpp"
#include <numa.h>
#include <vector>
#include "numa/hwloc_utils.hpp"
#include "numa/numa_mem.hpp"

hwlocUtils hwlocobj{};

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "usage: ./bw <mem_numa_node> <cuda_device> <concurrent>\n";
        return 1;
    }

    int mem_node = atoi(argv[1]);
    int cuda_dev = atoi(argv[2]);
    bool concurrent;

    if (strcmp(argv[3],"true") == 0) {
        concurrent = true;
    }
    else if (strcmp(argv[3],"false") == 0) {
        concurrent = false;
    }
    else {
        std::cerr << "invalid argument 3, it must be true or false" << std::endl;
        exit(-1);
    }

    hwlocobj.bindThreadToNUMA(mem_node);
    hwlocobj.bindMemToNUMA(mem_node);

    gpuSetDevice(cuda_dev);

    size_t gigabytes = 15;
    const size_t bytes = gigabytes * 1024 * 1024 * 1024;

    numaMem h2dmem(mem_node, bytes);
    numaMem d2hmem(mem_node, bytes);

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
    

    // ---------------- Streams ----------------
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

    std::cout << "Numa node " << mem_node << " <-> cuda device " << cuda_dev << ", concurrent = " << concurrent << std::endl;
    std::cout << "H2D: " << gigabytes / (totmsh2d/1e3) << " GBps" << std::endl;
    std::cout << "D2H: " << gigabytes / (totmsd2h/1e3) << " GBps" << std::endl;

    // ---------------- Cleanup ----------------
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
    
    return 0;
}
