#include "hwloc_utils.hpp"
#include <hwloc.h>
#include <iostream>

hwlocUtils::hwlocUtils() {
    hwloc_topology_init(&topo);
    hwloc_topology_load(topo);
}

void hwlocUtils::bindThreadToNUMA(int node) {
    hwloc_obj_t req_numa =
        hwloc_get_obj_by_type(
            topo, HWLOC_OBJ_NUMANODE, node);

    hwloc_set_cpubind(
        topo,
        req_numa->cpuset,
        HWLOC_CPUBIND_THREAD
    );
}

void hwlocUtils::verifyNumaPlacement(void* ptr, size_t bytes) {
    hwloc_nodeset_t ns = hwloc_bitmap_alloc();
    hwloc_membind_policy_t policy;

    int err = hwloc_get_area_membind(topo, ptr, bytes, ns, &policy, HWLOC_MEMBIND_BYNODESET);
    if (err) {
        std::cerr << "hwloc_get_area_membind failed: " << strerror(errno) << std::endl;
    } else {
        char buf[128];
        hwloc_bitmap_snprintf(buf, sizeof(buf), ns);
        std::cout << "Host buffer NUMA nodeset: " << buf << std::endl;
    }
    hwloc_bitmap_free(ns);
}

void hwlocUtils::bindMemToNUMA(int node) {
    hwloc_obj_t req_numa =
        hwloc_get_obj_by_type(
            topo, HWLOC_OBJ_NUMANODE, node);

    hwloc_set_membind(
        topo,
        req_numa->nodeset,
        HWLOC_MEMBIND_BIND,
        HWLOC_MEMBIND_THREAD
    );
}