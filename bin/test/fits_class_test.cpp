#include <iostream>
#include "fits/fits.hpp"
#include <memory>
#include <new>

struct AlignedDeleter {
    std::size_t align;
    void operator()(unsigned char* p) const noexcept {
        ::operator delete(p, std::align_val_t{align}); // matches allocation
    }
};

int main() {
    const char* filename = "/scratch/panchal/G0057_1255444104_00:30:27.42_+04:51:39.73_ch109-132_0001.fits";
    Fits obj(filename);

    std::unique_ptr<unsigned char, AlignedDeleter> buf(
    static_cast<unsigned char*>(
        ::operator new(obj.file_size_aligned(), std::align_val_t{4096})
    ),
    AlignedDeleter{4096});

    obj.setAlignedFileSizeBuffer(buf.get());

    obj.extractDataDirect();
    return 0; 
}