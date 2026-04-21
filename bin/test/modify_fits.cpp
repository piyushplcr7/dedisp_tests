#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include "dataHandlers/fits/fits.hpp"
#include "dataHandlers/datafile.hpp"

// Rewrites one or more FITS files so that data_scl, data_offs, data_wts
// produce the same 32-bit reduction as the GPU 8-bit unpack formula:
//   ((float)val - 127.5f) / nchan
//
// Usage:  modify_fits <input1.fits> [input2.fits ...]
// Output: <input1>_modified.fits  etc.

static std::string make_output_name(const char* input) {
    std::string s(input);
    auto dot = s.rfind('.');
    if (dot != std::string::npos)
        return s.substr(0, dot) + "_modified" + s.substr(dot);
    return s + "_modified";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input1.fits> [input2.fits ...]"
                  << std::endl;
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        const char* infile = argv[i];
        std::string outfile = make_output_name(infile);

        std::cout << "[" << i << "/" << (argc - 1) << "] " << infile
                  << " -> " << outfile << std::endl;

        Fits fits(infile);
        fits.writeModifiedFits(outfile.c_str());
    }

    return 0;
}
