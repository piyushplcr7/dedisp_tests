#ifndef HWLOCUTILSHPP
#define HWLOCUTILSHPP

#include <hwloc.h>

class hwlocUtils {
private:
    hwloc_topology_t topo;
public:
    hwlocUtils();

    void bindThreadToNUMA(int node);
    void bindMemToNUMA(int node);
    void verifyNumaPlacement(void* ptr, size_t bytes);
};

#endif