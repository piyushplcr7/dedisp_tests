#include <iostream>
#include "fits/fits.hpp"
#include <vector>
#include <string>
#include <numa.h>
#include <numaif.h>
#include <memory>
#include <chrono>

int main() {
    std::vector<std::string> filenames;
    filenames.push_back(std::string("/tmp/panchal/1.fits"));
    //filenames.push_back(std::string("/scratch/panchal/2.fits"));

    // Test 1: 
    // Allocate aligned buffer in numa node 1 (IB) and read fits there
    std::cout << "Constructing Fits object from " << filenames[0].c_str() << std::endl;
    Fits obj(filenames[0].c_str());
    std::cout << "constructed fits" << std::endl;

    std::cout << "allocating aligned buf" << std::endl;
    std::unique_ptr<unsigned char, void (*)(unsigned char*)> alignedBuf(
        static_cast<unsigned char*>( ::operator new(obj.fileSizeAligned() , std::align_val_t(4096))),
        [] (unsigned char* x) { ::operator delete(x, std::align_val_t(4096)); }
    );
    std::cout << "allocated buffer" << std::endl;

    //size_t chunksize = 1073741824; 
    size_t chunksize = 2147479552;
    //size_t chunksize = obj.naxis1();

    std::cout << "extracting data direct" << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    obj.extractDataDirect(alignedBuf.get(), chunksize);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end-start;
    std::cout << "Reading took " << diff.count() << " seconds at speed = " << obj.fileSizeAligned()/(1<<20)/diff.count() << " MB/s" << std::endl;

    return 0;
}