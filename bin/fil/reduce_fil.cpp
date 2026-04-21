#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <string>
#include "dataHandlers/fil/fil.hpp"
#include "dataHandlers/datafile.hpp"

// Reads one or more .fil files, applies the GPU-matching 8-bit unpack
// formula ((val - 127.5f) / nchan) to produce 32-bit float data,
// and writes the result as a raw binary .dat file.
//
// Usage:  reduce_fil <input1.fil> [input2.fil ...]
// Output: <input1>.dat  etc.

static std::string make_output_name(const char* input) {
    std::string s(input);
    auto dot = s.rfind('.');
    if (dot != std::string::npos)
        return s.substr(0, dot) + ".dat";
    return s + ".dat";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input1.fil> [input2.fil ...]"
                  << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        const char* infile = argv[i];
        std::string outfile = make_output_name(infile);

        std::cout << "[" << i << "/" << (argc - 1) << "] " << infile
                  << " -> " << outfile << std::endl;

        Fil fil(infile);
        fil.printInfo();

        // Allocate aligned buffer for O_DIRECT read
        unsigned char* buf = static_cast<unsigned char*>(
            std::aligned_alloc(4096, fil.fileSizeAligned()));
        if (!buf) {
            std::cerr << "Failed to allocate aligned buffer for " << infile
                      << std::endl;
            return 1;
        }

        fil.extractDataDirect(buf, HALF_MAX_CHUNKSIZE);

        // Allocate float output buffer
        size_t nelem = fil.getNumElements();
        float* data = static_cast<float*>(
            std::malloc(nelem * sizeof(float)));
        if (!data) {
            std::cerr << "Failed to allocate data buffer for " << infile
                      << std::endl;
            std::free(buf);
            return 1;
        }

        // Reduce: applies (val - 127.5f) / nchan per sample
        fil.reduceData(reinterpret_cast<unsigned char*>(data), 32, 0, 1);

        // Write raw float data
        FILE* fp = fopen(outfile.c_str(), "wb");
        if (!fp) {
            std::cerr << "Cannot open " << outfile << " for writing"
                      << std::endl;
            std::free(data);
            std::free(buf);
            return 1;
        }
        fwrite(data, sizeof(float), nelem, fp);
        fclose(fp);

        std::cout << "Wrote " << outfile << " (" << nelem << " floats, "
                  << nelem * sizeof(float) << " bytes)" << std::endl;

        std::free(data);
        std::free(buf);
    }

    return 0;
}
