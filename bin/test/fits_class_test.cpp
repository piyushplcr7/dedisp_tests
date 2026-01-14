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
    const char* filename = "/scratch/panchal/1.fits";
    Fits obj(filename);

    std::cout << "obj.file_size_aligned() = " << obj.fileSizeAligned() << std::endl;

    size_t numelements = obj.getNumElements();
    std::cout << "obj.getNumElements() = " << numelements << std::endl;

    unsigned char* buf = static_cast<unsigned char*> (std::aligned_alloc(4096, obj.fileSizeAligned()));

    /* std::unique_ptr<unsigned char, AlignedDeleter> buf(
    static_cast<unsigned char*>(
        ::operator new(obj.file_size_aligned(), std::align_val_t{4096})
    ),
    AlignedDeleter{4096}); */

    float* data = static_cast<float*> (std::malloc(obj.getNumElements() * sizeof(float)));

    obj.setAlignedFileSizeBuffer(buf);
    obj.setDataBuffer(data);

    std::cout << "extracting data" << std::endl;
    obj.extractDataDirect();
    std::cout << "extracting data finished" << std::endl;

    std::cout << "reduce data" << std::endl;
    obj.reduceData();
    std::cout << "reducing data finished" << std::endl;

    std::cout << "freeing data buffer" << std::endl;
    std::free(data);
    std::cout << "freed data buffer" << std::endl;
    
    std::cout << "freeing buf" << std::endl;
    std::free(buf);
    std::cout << "freed buf" << std::endl;

    return 0; 
}