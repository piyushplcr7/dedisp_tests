#ifndef NUMAMEMHPP
#define NUMAMEMHPP

#include <numa.h>

class numaMem {
private:
    void* buf_;
    size_t bytes_;
public:
    numaMem(int node, size_t bytes);
    ~numaMem();

    const void* get() const{ return buf_; }
    void* get() { return buf_; }
    size_t size() const{ return bytes_; }
    void touchPages();
};

#endif