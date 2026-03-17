#ifndef FILHPP
#define FILHPP

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include "../fits/matrix_view.hpp"

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
class Fil {

  public:
    // Constructor — reads the SIGPROC header and computes all metadata.
    Fil(const char* filename);

    // ----- Getters (read-only API, mirrors Fits) -----
    int         nchan()        const noexcept { return nchans_read_; }
    double      f_lo()         const { return freqs_.front(); }   // lowest freq
    double      f_hi()         const { return freqs_.back();  }   // highest freq
    // Total time samples (after optional downsampling)
    size_t      dimTime(unsigned int downsamp = 1) const { return (size_t)nsblk_ / downsamp; }
    size_t      fileSizeAligned() const noexcept { return file_size_aligned_; }
    size_t      fileSize()     const { return file_size_; }
    float*      data()         const noexcept { return data_; }
    // naxis1: row width in bytes — not applicable for filterbank; returns 0.
    int         naxis1()       const noexcept { return naxis1_; }
    // nsblk: treated as the total number of time samples (single "subint").
    int         nsblk()        const noexcept { return nsblk_; }
    int         Tobs()         const noexcept { return (int)(tbin_ * nsblk_); }
    // f0: highest frequency channel (reference for dedispersion)
    double      f0()           const noexcept { return freqs_.back(); }
    const char* telescope()    const noexcept { return telescope_name_; }
    const char* instrument()   const noexcept { return instrument_; }
    const char* objectName()   const noexcept { return object_name_; }
    // rightAscension / declination: formatted as "HH:MM:SS.SSSSSS" / "±DD:MM:SS.SSSSSS"
    const char* rightAscension() const noexcept { return ra_; }
    char*       rightAscension()              { return ra_; }
    const char* declination()   const noexcept { return dec_; }
    char*       declination()                  { return dec_; }
    const char* observer()      const noexcept { return observer_; }
    const char* projid()        const noexcept { return projid_; }
    // epoch: tstart in MJD
    double      epoch()         const noexcept { return epoch_; }
    const char* dateobs()       const noexcept { return dateobs_; }
    // obs: 2-char TEMPO observatory code
    const char* obs()           const noexcept { return obs_; }
    char*       obs()                          { return obs_; }
    const char* ephem()         const noexcept { return ephem_; }
    char*       ephem()                        { return ephem_; }

    // Channel frequency step (negative for descending-frequency filterbanks).
    // Computed as (freqs_[0] - freqs_[nchan-1]) / (nchan-1) = foff_.
    double      ddf()          const noexcept {
        return (freqs_[0] - freqs_[nchans_read_ - 1]) / (nchans_read_ - 1);
    }
    double      bw()           const noexcept { return -ddf() * nchans_read_; }
    double      sampletime(unsigned int downsamp = 1) const noexcept { return tbin_ * downsamp; }

    void        printInfo()    const noexcept;

    // Channel frequencies in ascending order (same convention as Fits::freqs_).
    const std::vector<double>& freqs() const noexcept { return freqs_; }

    // Read the file into an O_DIRECT-aligned buffer in chunks.
    // Mirrors Fits::getDataFromRows.
    long getDataFromRows(int fd, unsigned char* table_data, long chunksize,
                         long bytes_to_read, int fd_nodirect);

    // Read the entire file (header + data) into aligned_filesize_buffer_.
    // Must call setAlignedFileSizeBuffer() first.
    void extractDataDirect(size_t chunksize = 1073741824);

    // Reduce the raw bytes in aligned_filesize_buffer_ into the float data_ buffer.
    // poln: polarisation index (only 0 is supported for nif = 1 filterbanks).
    // downsamp: time downsampling factor; nsblk_ must be divisible by downsamp.
    void reduceData(int poln = 0, unsigned int downsamp = 1);

    void setAlignedFileSizeBuffer(unsigned char* buf) { aligned_filesize_buffer_ = buf; }
    void setDataBuffer(float* data) { data_ = data; }
    void setVerbosity(int verbosity) { verbose_ = verbosity; }

    size_t getNumElements(unsigned int downsamp = 1) const {
        return dimTime(downsamp) * nchans_read_;
    }
    size_t getDataSize(unsigned int downsamp = 1) const {
        return getNumElements(downsamp) * sizeof(float);
    }

    matrixView<float>&       dataView()       { return dataView_; }
    const matrixView<float>& dataView() const { return dataView_; }

    const char* getFilename() const { return filename_; }

    ~Fil() {}

  private:
    const char* filename_;

    // Metadata strings
    char telescope_name_[40]  = {};   // derived from telescope_id
    char instrument_[100]     = {};   // derived from machine_id
    char object_name_[100]    = {};   // source_name
    char ra_[50]              = {};   // "HH:MM:SS.SSSSSS"
    char dec_[50]             = {};   // "±DD:MM:SS.SSSSSS"
    char observer_[100]       = {};   // not in filterbank header — "Unknown"
    char projid_[100]         = {};   // not in filterbank header — "Unknown"
    char dateobs_[100]        = {};   // derived from tstart
    char ephem_[10]           = "DE405";
    char obs_[3]              = {};   // TEMPO observatory code
    char outscope_[40]        = {};   // TEMPO scope name (internal)

    // Timing / frequency
    double tbin_              = 0.0;  // tsamp (seconds)
    double epoch_             = 0.0;  // tstart (MJD)
    double fch1_              = 0.0;  // centre frequency of channel 0 (MHz)
    double foff_              = 0.0;  // channel step (MHz, negative for descending)
    double src_raj_           = 0.0;  // SIGPROC: HHMMSS.SS
    double src_dej_           = 0.0;  // SIGPROC: ±DDMMSS.SS

    // Dimensions
    int    nchans_read_       = 0;
    // nsblk_ holds the total number of time samples (the entire file is treated
    // as a single "subintegration" so that dimTime() = nsblk_ / downsamp).
    int    nsblk_             = 0;
    int    naxis1_            = 0;    // not applicable for filterbank
    int    naxis2_            = 1;    // always 1 (one "subint")
    int    nbin_              = 1;    // always 1 (search-mode data)
    int    npol_              = 1;    // nif from header
    int    nbit_              = 8;
    int    telescope_id_      = 0;
    int    machine_id_        = 0;
    int    verbose_           = 0;

    // File layout
    size_t headersize_        = 0;    // byte length of SIGPROC header
    size_t buffersize_        = 0;    // nsamp * nchan * nbit/8
    size_t file_size_         = 0;
    size_t file_size_aligned_ = 0;   // aligned to 4096

    // Channel frequencies in ascending order:
    //   freqs_[c] = fch1_ + (nchans_read_ - 1 - c) * foff_
    std::vector<double> freqs_;

    float*         data_                   = nullptr;
    unsigned char* aligned_filesize_buffer_ = nullptr;
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
