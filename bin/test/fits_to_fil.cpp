// fits_to_fil: Custom FITS -> filterbank converter preserving exact raw 8-bit bytes.
//
// Uses a digifil-produced .fil as a header template, but:
//   (1) replaces the data section with the exact poln-0 raw bytes extracted
//       from the FITS binary table via reduceBinaryTable8, AND
//   (2) patches SIGPROC header fields (fch1, foff, tsamp, tstart) with values
//       derived from the FITS file, so that Plan parameters (f0, ddf, dt,
//       epoch) match exactly between FITS and FIL paths.
//
// Why (2): digifil typically writes fch1 shifted by +foff/2 (edge vs. centre
// convention), producing DM-delay differences at non-zero DMs. By injecting
// the FITS-derived values we guarantee matching dedispersion results.
//
// Usage: fits_to_fil <input.fits> <template.fil> <output.fil>

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <vector>
#include "dataHandlers/fits/fits.hpp"
#include "dataHandlers/fil/fil.hpp"
#include "dataHandlers/datafile.hpp"

// Locate a SIGPROC key [int32=len][key_bytes] inside a header buffer and
// overwrite the 8-byte double value that follows. Returns true if patched.
static bool patch_sigproc_double(std::vector<unsigned char>& hdr,
                                 const char* key, double new_val,
                                 double* old_val_out = nullptr) {
    const size_t key_len = std::strlen(key);
    if (hdr.size() < 4 + key_len + sizeof(double)) return false;

    for (size_t i = 0; i + 4 + key_len + sizeof(double) <= hdr.size(); ++i) {
        int32_t len;
        std::memcpy(&len, hdr.data() + i, sizeof(int32_t));
        if (len != (int32_t)key_len) continue;
        if (std::memcmp(hdr.data() + i + 4, key, key_len) != 0) continue;

        if (old_val_out) {
            std::memcpy(old_val_out, hdr.data() + i + 4 + key_len, sizeof(double));
        }
        std::memcpy(hdr.data() + i + 4 + key_len, &new_val, sizeof(double));
        return true;
    }
    return false;
}

static void patch_and_report(std::vector<unsigned char>& hdr,
                             const char* key, double new_val) {
    double old_val = 0.0;
    bool ok = patch_sigproc_double(hdr, key, new_val, &old_val);
    if (!ok) {
        std::cerr << "WARNING: SIGPROC key '" << key
                  << "' not found in template header; skipping patch" << std::endl;
    } else {
        std::cout << "  patched " << key << ": "
                  << old_val << " -> " << new_val
                  << "  (delta=" << (new_val - old_val) << ")" << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.fits> <template.fil> <output.fil>" << std::endl;
        return 1;
    }

    const char* fits_path = argv[1];
    const char* tmpl_path = argv[2];
    const char* out_path  = argv[3];

    // --- FITS: load and extract poln 0, 8-bit raw ---
    Fits fits(fits_path);
    fits.printInfo();

    unsigned char* fits_iobuf = static_cast<unsigned char*>(
        std::aligned_alloc(4096, fits.fileSizeAligned()));
    fits.extractDataDirect(fits_iobuf, HALF_MAX_CHUNKSIZE);

    size_t fits_nelem = fits.getNumElements();
    unsigned char* fits_data = static_cast<unsigned char*>(std::calloc(fits_nelem, 1));
    fits.reduceData(fits_data, 8, 0, 1);

    // --- Template FIL: read header ---
    Fil tmpl(tmpl_path);
    tmpl.printInfo();

    size_t hdr_size = tmpl.headerSize();

    // Read the raw header bytes from the template file
    FILE* tf = fopen(tmpl_path, "rb");
    if (!tf) {
        std::cerr << "ERROR: cannot open template " << tmpl_path << std::endl;
        return 1;
    }
    std::vector<unsigned char> hdr(hdr_size);
    if (fread(hdr.data(), 1, hdr_size, tf) != hdr_size) {
        std::cerr << "ERROR: short read on template header" << std::endl;
        fclose(tf);
        return 1;
    }
    fclose(tf);

    // --- Sanity checks ---
    if (fits.nchan() != tmpl.nchan()) {
        std::cerr << "ERROR: nchan mismatch: FITS=" << fits.nchan()
                  << " FIL=" << tmpl.nchan() << std::endl;
        return 1;
    }

    std::cout << "\nMetadata comparison (before patch):" << std::endl;
    std::cout.precision(15);
    std::cout << "  FITS  f0=" << fits.f0() << "  ddf=" << fits.ddf()
              << "  tsamp=" << fits.sampletime()
              << "  tstart=" << fits.epoch() << std::endl;
    std::cout << "  FIL   f0=" << tmpl.f0() << "  ddf=" << tmpl.ddf()
              << "  tsamp=" << tmpl.sampletime()
              << "  tstart=" << tmpl.epoch() << std::endl;

    if (fits.dimTime() != tmpl.dimTime()) {
        std::cout << "NOTE: nsamp differs (FITS=" << fits.dimTime()
                  << " vs template=" << tmpl.dimTime()
                  << "). Output will have FITS nsamp." << std::endl;
    }

    // --- Patch SIGPROC header fields to match FITS-derived values ---
    // fch1  = freqs_.back() (highest channel freq, matches Fil convention
    //         since Fil::freqs_[nchan-1] = fch1)
    // foff  = ddf() (negative step, matches Fil::foff_ convention)
    // tsamp = tbin
    // tstart = epoch (MJD)
    std::cout << "\nPatching SIGPROC header with FITS-derived values:" << std::endl;
    patch_and_report(hdr, "fch1",   fits.f0());
    patch_and_report(hdr, "foff",   fits.ddf());
    patch_and_report(hdr, "tsamp",  fits.sampletime());
    patch_and_report(hdr, "tstart", fits.epoch());

    // --- Write output: patched template header + FITS raw data ---
    FILE* out = fopen(out_path, "wb");
    if (!out) {
        std::cerr << "ERROR: cannot open output " << out_path << std::endl;
        return 1;
    }
    fwrite(hdr.data(), 1, hdr_size, out);
    fwrite(fits_data, 1, fits_nelem, out);
    fclose(out);

    std::cout << "\nWrote " << out_path << ": "
              << hdr_size << " header + " << fits_nelem << " data bytes" << std::endl;

    std::free(fits_data);
    std::free(fits_iobuf);

    return 0;
}
