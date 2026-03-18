#include <iostream>
#include "datafile.hpp"
#include "fits.hpp"
#include "fil.hpp"
#include <memory>
#include <new>

struct AlignedDeleter {
    std::size_t align;
    void operator()(unsigned char* p) const noexcept {
        ::operator delete(p, std::align_val_t{align}); // matches allocation
    }
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <filename.fits>" << std::endl;
        return 1;
    }

    const char* filename = argv[1];

    dataFile *obj;

    // Decide which dataFile subclass to instantiate based on file extension
    std::string fname(filename);
    if (fname.size() >= 5 && fname.substr(fname.size() - 5) == ".fits") {
        obj = new Fits(filename);
    }
    else if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".fil") {
        obj = new Fil(filename);
    }
    else {
        std::cerr << "Unsupported file type! Please provide a .fits or .fil file." << std::endl;
        return 1;
    }

    std::cout << "obj.file_size_aligned() = " << obj->fileSizeAligned() << std::endl;

    size_t numelements = obj->getNumElements();
    std::cout << "obj.getNumElements() = " << numelements << std::endl;

    unsigned char* buf = static_cast<unsigned char*> (std::aligned_alloc(4096, obj->fileSizeAligned()));

    /* std::unique_ptr<unsigned char, AlignedDeleter> buf(
    static_cast<unsigned char*>(
        ::operator new(obj->fileSizeAligned(), std::align_val_t{4096})
    ),
    AlignedDeleter{4096}); */

    float* data = static_cast<float*> (std::malloc(obj->getNumElements() * sizeof(float)));

    std::cout << "extracting data" << std::endl;
    obj->extractDataDirect(buf, HALF_MAX_CHUNKSIZE);
    std::cout << "extracting data finished" << std::endl;

    std::cout << "reduce data" << std::endl;
    obj->reduceData((unsigned char*)data, 32, 0, 1);
    std::cout << "reducing data finished" << std::endl;

    std::cout << "freeing data buffer" << std::endl;
    std::free(data);
    std::cout << "freed data buffer" << std::endl;
    
    std::cout << "freeing buf" << std::endl;
    std::free(buf);
    std::cout << "freed buf" << std::endl;

    return 0; 
}