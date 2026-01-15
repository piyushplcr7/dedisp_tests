#include <iostream>
#include <string>
#include <vector>
#include <hwloc.h>
#include <memory>
#include <chrono>
#include <thread>
#include "fits/fits.hpp"

void extractOnNode(int node, Fits& fits) {
    // Get topology info
    hwloc_topology_t topo;
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);

    // Allocate aligned memory on numa node
    hwloc_obj_t req_numa =
        hwloc_get_obj_by_type(
            topo, HWLOC_OBJ_NUMANODE, node);

    hwloc_set_cpubind(
        topo,
        req_numa->cpuset,
        HWLOC_CPUBIND_THREAD
    );

    hwloc_set_membind(
        topo,
        req_numa->nodeset,
        HWLOC_MEMBIND_BIND,
        HWLOC_MEMBIND_THREAD
    );

    // Allocation will be on node
    std::unique_ptr<unsigned char, void (*)(unsigned char*)> alignedBuf(
        static_cast<unsigned char*>( ::operator new(fits.fileSizeAligned() , std::align_val_t(4096))),
        [] (unsigned char* x) { ::operator delete(x, std::align_val_t(4096)); }
    );

    fits.setAlignedFileSizeBuffer(alignedBuf.get());

    auto start = std::chrono::high_resolution_clock::now();
    fits.extractDataDirect(fits.naxis1());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Extracted fits on node " << node << " at " << (double)fits.fileSize()/(1<<20)/diff.count() << " MBps" << std::endl; 
}

void parallelLoadTest() {
    Fits fits1("/scratch/panchal/1.fits");
    Fits fits2("/scratch/panchal/2.fits");
    Fits fits3("/scratch/panchal/3.fits");

    auto start = std::chrono::high_resolution_clock::now();
    std::thread t1(extractOnNode, 0, std::ref(fits1));
    std::thread t2(extractOnNode, 1, std::ref(fits2));
    std::thread t3(extractOnNode, 2, std::ref(fits3));

    t1.join();
    t2.join();
    t3.join();
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;

    std::cout << "Effective read speed: " << ((double) (fits1.fileSize() + fits2.fileSize()))/(1<<30)/diff.count() << " GBps" << std::endl;
}

void singleLoadTest(int node) {
    Fits fits("/scratch/panchal/1.fits");
    extractOnNode(node, fits);
}

int main(int argc, char** argv) {
    int node;
    if (argc > 1) {
        node = atoi(argv[1]);
    }

    //singleLoadTest(node);
    parallelLoadTest();

    return 0;
}


/*
* Results on kuma

(base) [panchal@kh016 dedisp_tests]$ ./build/bin/test/parFitsLoad 
Extracted fits on node 0 at 4064.47 MBps
Extracted fits on node 1 at 4074.83 MBps
Effective read speed: 7.80543 GBps

(base) [panchal@kh016 dedisp_tests]$ ./build/bin/test/parFitsLoad 
Extracted fits on node 2 at 3951.75 MBps
Extracted fits on node 1 at 3904.68 MBps
Extracted fits on node 0 at 3820.88 MBps
Effective read speed: 7.39023 GBps


*/