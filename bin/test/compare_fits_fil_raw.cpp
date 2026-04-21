#include <iostream>
#include <cstdlib>
#include <cstring>
#include "dataHandlers/fits/fits.hpp"
#include "dataHandlers/fil/fil.hpp"
#include "dataHandlers/datafile.hpp"

// Compare the raw 8-bit data extracted from a FITS file and a FIL file.
// Both are reduced via their respective reduceData(..., 8, ...) paths.
//
// Usage: compare_fits_fil_raw <file.fits> <file.fil>

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file.fits> <file.fil>" << std::endl;
        return 1;
    }

    const char* fits_path = argv[1];
    const char* fil_path  = argv[2];

    // --- FITS ---
    Fits fits(fits_path);
    fits.printInfo();

    unsigned char* fits_iobuf = static_cast<unsigned char*>(
        std::aligned_alloc(4096, fits.fileSizeAligned()));
    fits.extractDataDirect(fits_iobuf, HALF_MAX_CHUNKSIZE);

    size_t fits_nelem = fits.getNumElements();
    unsigned char* fits_data = static_cast<unsigned char*>(std::calloc(fits_nelem, 1));
    fits.reduceData(fits_data, 8, 0, 1);

    // --- FIL ---
    Fil fil(fil_path);
    fil.printInfo();

    unsigned char* fil_iobuf = static_cast<unsigned char*>(
        std::aligned_alloc(4096, fil.fileSizeAligned()));
    fil.extractDataDirect(fil_iobuf, HALF_MAX_CHUNKSIZE);

    size_t fil_nelem = fil.getNumElements();
    unsigned char* fil_data = static_cast<unsigned char*>(std::calloc(fil_nelem, 1));
    fil.reduceData(fil_data, 8, 0, 1);

    // --- Compare ---
    std::cout << "\n=== Comparing raw 8-bit data ===" << std::endl;
    std::cout << "  FITS elements: " << fits_nelem << std::endl;
    std::cout << "  FIL  elements: " << fil_nelem << std::endl;

    size_t n = std::min(fits_nelem, fil_nelem);
    size_t mismatches = 0;
    size_t first_mismatch = 0;

    for (size_t i = 0; i < n; ++i) {
        if (fits_data[i] != fil_data[i]) {
            if (mismatches == 0) first_mismatch = i;
            ++mismatches;
        }
    }

    if (mismatches == 0 && fits_nelem == fil_nelem) {
        std::cout << "PASS: raw 8-bit data is identical (" << n << " bytes)" << std::endl;
    } else {
        std::cout << "FAIL:" << std::endl;
        if (fits_nelem != fil_nelem)
            std::cout << "  Length mismatch: " << fits_nelem << " vs " << fil_nelem << std::endl;
        std::cout << "  Mismatched bytes: " << mismatches << " / " << n << std::endl;
        if (mismatches > 0) {
            std::cout << "  First mismatch at index " << first_mismatch
                      << ": FITS=" << (int)fits_data[first_mismatch]
                      << " FIL=" << (int)fil_data[first_mismatch] << std::endl;
            // Show a few more
            size_t shown = 0;
            for (size_t i = first_mismatch; i < n && shown < 10; ++i) {
                if (fits_data[i] != fil_data[i]) {
                    std::cout << "    [" << i << "] FITS=" << (int)fits_data[i]
                              << " FIL=" << (int)fil_data[i] << std::endl;
                    ++shown;
                }
            }
        }
    }

    std::free(fits_data);
    std::free(fil_data);
    std::free(fits_iobuf);
    std::free(fil_iobuf);

    return mismatches > 0 ? 1 : 0;
}
