#ifndef FILHPP
#define FILHPP

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include "matrix_view.hpp"
#include "datafile.hpp"

/*
 * Fil: represents a single SIGPROC filterbank file.
 *
 * Mirrors the Fits class interface so that both can be used
 * interchangeably in a unified container (filecontainer).
 *
 * Frequency / channel ordering convention (same as Fits):
 *   freqs_[0]       = lowest channel frequency
 *   freqs_[nchan-1] = highest channel frequency  (= fch1)
 *   f0()  = freqs_.back()  = fch1  (highest freq, reference channel)
 *   ddf() = foff           (negative step, computed from freqs_ formula)
 *
 * Data buffer layout (same as Fits, matches what the Plan kernel expects):
 *   data_[t * nchan + c],  c = 0 → fch1 (highest freq), c = nchan-1 → lowest
 *   i.e. the raw SIGPROC byte order is preserved — no channel reversal.
 *
 * Supported: nbit = 8, nif = 1 (polarisation index 0).
 * Multi-IF / multi-bit support is noted where assumptions are made.
 */
class Fil : public dataFile {

  public:
    // Constructor — reads the SIGPROC header and computes all metadata.
    Fil(const char* filename);

    // ----- Getters (read-only API, mirrors Fits) -----
    // Total time samples (after optional downsampling)
    size_t      dimTime(unsigned int downsamp = 1) const override { return (size_t)dimTime_ / downsamp; }
    int         Tobs()         const noexcept { return (int)(tbin_ * dimTime_); }

    void        printInfo() const noexcept override;

    // Read the file into an O_DIRECT-aligned buffer in chunks.
    // Mirrors Fits::getDataFromRows.
    void reduceData(unsigned char* outBuf, int out_bits=32, int poln = 0, unsigned int downsamp = 1) override;
    bool checkDownsamp(unsigned int downsamp) const override { return dimTime_ % downsamp == 0; }

    size_t getDataSize(unsigned int downsamp = 1) const {
        return getNumElements(downsamp) * sizeof(float);
    }

    matrixView<float>&       dataView()       { return dataView_; }
    const matrixView<float>& dataView() const { return dataView_; }

    // Write the read data back to disk as a SIGPROC filterbank file.
    // Mirrors the write logic in dedisp_fil.
    void writeToDisk(const char* outfile) const;

    ~Fil() {}

  private:

    // Fil-specific dimensions
    int    dimTime_   = 0;   // total time samples

    // File layout
    size_t headersize_ = 0;   // byte length of SIGPROC header
    size_t buffersize_ = 0;   // nsamp * nchan * nbit/8

    matrixView<float> dataView_;

    // Internal helpers
    void parseHeader();
    void rajToString(double raj, char* out, size_t sz);
    void dejToString(double dej, char* out, size_t sz);

    // Raw data reduction helpers (analogous to Fits::reduceBinaryTable)
    void reduceRawData(const unsigned char* raw, float* data,
                       int nchan, size_t nsamp);
    void reduceRawDataDownSamp(const unsigned char* raw, float* data,
                               int nchan, size_t nsamp, unsigned int downsamp);

    static const char* telescopeName(int id);
    static const char* machineName(int id);
};

#endif // FILHPP
