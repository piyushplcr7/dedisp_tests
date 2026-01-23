#include "numa_mem.hpp"
#include <numa.h>

numaMem::numaMem(int node, size_t bytes):bytes_(bytes) {
    buf_ = numa_alloc_onnode(bytes, node);
}

void numaMem::touchPages() {
    volatile char *p = (volatile char *)buf_;
    const size_t page = 4096;

    for (size_t i = 0; i < bytes_; i += page) {
        p[i] = 0;
    }
}

numaMem::~numaMem() {
    numa_free(buf_, bytes_);
}