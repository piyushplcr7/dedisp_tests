#include <iostream>
#include <cstdlib>
#include "dataHandlers/fil/fil.hpp"
#include "dataHandlers/datafile.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input.fil> <output.dat>" << std::endl;
        return 1;
    }

    const char* infile  = argv[1];
    const char* outfile = argv[2];

    Fil fil(infile);
    fil.printInfo();

    unsigned char* buf = static_cast<unsigned char*>(
        std::aligned_alloc(4096, fil.fileSizeAligned()));
    if (!buf) {
        std::cerr << "Failed to allocate aligned buffer" << std::endl;
        return 1;
    }

    std::cout << "Extracting data..." << std::endl;
    fil.extractDataDirect(buf, HALF_MAX_CHUNKSIZE);

    float* data = static_cast<float*>(std::malloc(fil.getNumElements() * sizeof(float)));
    if (!data) {
        std::cerr << "Failed to allocate data buffer" << std::endl;
        std::free(buf);
        return 1;
    }

    std::cout << "Reducing data..." << std::endl;
    fil.reduceData(reinterpret_cast<unsigned char*>(data), 32, 0, 1);

    std::cout << "Writing data to disk..." << std::endl;
    fil.writeToDisk(outfile);

    std::free(data);
    std::free(buf);

    return 0;
}
