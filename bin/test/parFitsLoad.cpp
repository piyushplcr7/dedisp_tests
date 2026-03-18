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

    auto start = std::chrono::high_resolution_clock::now();
    fits.extractDataDirect(alignedBuf.get(), fits.naxis1());
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << "Extracted fits on node " << node << " at " << (double)fits.fileSize()/(1<<20)/diff.count() << " MBps" << std::endl; 
}

void parallelLoadTest(std::vector<int> nodes) {
    int parFiles = nodes.size();
    std::vector<Fits> fits;
    fits.reserve(4);

    fits.emplace_back("/scratch/panchal/1.fits");
    fits.emplace_back("/scratch/panchal/2.fits");
    fits.emplace_back("/scratch/panchal/3.fits");
    fits.emplace_back("/scratch/panchal/4.fits");

    std::vector<std::thread> threads(parFiles);

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0 ; i < parFiles ; ++i) {
        threads[i] = std::thread(extractOnNode, nodes[i], std::ref(fits[i]));
    }

    for (int i = 0 ; i < parFiles ; ++i) {
        threads[i].join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;

    size_t tot_size = 0;

    for (int i = 0 ; i < parFiles ; ++i) {
        tot_size += fits[i].fileSize();
    }

    std::cout << "Effective read speed: " << ((double) tot_size )/(1<<30)/diff.count() << " GBps" << std::endl;
}

int main(int argc, char** argv) {
    int node;
    if (argc > 5 or argc == 1) {
        std::cerr << "Specify 1-4 nodes to load fits on" << std::endl;
    }

    std::vector<int> nodes(argc-1);

    for (int i = 1 ; i < argc ; ++i) {
        nodes[i-1] = atoi(argv[i]);
    }
    
    parallelLoadTest(nodes);

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