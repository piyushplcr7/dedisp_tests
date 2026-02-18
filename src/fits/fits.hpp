#ifndef FITSHPP
#define FITSHPP

#include<vector>
#include<string>
#include <fitsio.h>
#include <byteswap.h>
#include <cstdint>
#include <string>
#include <iostream>
#include "matrix_view.hpp"

// Longest chunksize 2147479552 for direct read (linux read documentation)
#define MAX_CHUNKSIZE 2147479552
#define HALF_MAX_CHUNKSIZE 1073741824

inline void swap_endian_3floats(float &f1, float &f2, float &f3) {
  uint32_t *p1 = reinterpret_cast<uint32_t *>(&f1);
  uint32_t *p2 = reinterpret_cast<uint32_t *>(&f2);
  uint32_t *p3 = reinterpret_cast<uint32_t *>(&f3);

  *p1 = bswap_32(*p1);
  *p2 = bswap_32(*p2);
  *p3 = bswap_32(*p3);
}

class Fits {
  public:
    // Constructor with the filename
    Fits(const char* filename);

    // ----- Getters (read-only API) -----
    int         nchan()        const noexcept { return nchans_read_; }      // NCHAN
    double      f_lo()         const { return freqs_.front(); }
    double      f_hi()         const { return freqs_.back();  }
    size_t      dimTime(unsigned int downsamp=1)      const { return ((size_t)naxis2_ * nsblk_)/downsamp; }
    size_t      fileSizeAligned() const noexcept { return file_size_aligned_; }
    size_t      fileSize()     const { return file_size_; }
    float*      data()         const noexcept { return data_; }
    int         naxis1()       const noexcept {return naxis1_; }
    int         nsblk()        const noexcept {return nsblk_; }
    int         Tobs()         const noexcept {return tbin_ * (nsblk_ * naxis2_); }
    double      f0()           const noexcept {return freqs_.back(); }
    const char* telescope()    const noexcept { return telescope_name_; }
    const char* instrument()   const noexcept { return instrument_; }
    const char* objectName()   const noexcept { return object_name_; }
    const char* rightAscension() const noexcept { return ra_; }
    char*       rightAscension()              { return ra_; }
    const char* declination()   const noexcept { return dec_; }
    char*       declination()                  { return dec_; }
    const char* observer()      const noexcept { return observer_; }
    const char* projid()        const noexcept { return projid_; }
    double      epoch()         const noexcept { return epoch_; }
    const char* dateobs()       const noexcept { return dateobs_; }
    const char* obs()           const noexcept { return obs_; }
    char*       obs()                          { return obs_; }
    const char* ephem()         const noexcept { return ephem_; }
    char*       ephem()                        { return ephem_; }
    double      ddf()          const noexcept {
      return (freqs_[0] - freqs_[nchans_read_ - 1]) /(nchans_read_ - 1);
    }
    double      bw()           const noexcept {
      return -ddf() * nchans_read_;
    }
    double      sampletime(unsigned int downsamp=1)   const noexcept {return tbin_ * downsamp; }
    void        printInfo()      const noexcept;
    // Channel frequencies column (DAT_FREQ)
    const std::vector<double>& freqs() const noexcept { return freqs_; }

    long getDataFromRows(int fd, unsigned char *table_data, long chunksize, long bytes_to_read, int fd_nodirect);
    
    // Reduction functions to reduce the rawdata in the aligned_filesize_buffer_ to
    // data_ buffer
    void reduceBinaryTable(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width);

    void reduceBinaryTableDownSamp(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width, int downsamp);
    
    // Reading all the data from the fits file using direct IO
    // into the aligned_filesize_buffer_
    void extractDataDirect(size_t chunksize=1073741824);

    // Reduce the data into the data buffer. The first dimension in the reduced data is channels
    void reduceData(int poln=0, unsigned int downsamp=1);

    void setAlignedFileSizeBuffer(unsigned char* buf) { 
      aligned_filesize_buffer_ = buf; 
    }
    
    void setDataBuffer(float* data) { data_ = data; }

    void setVerbosity(int verbosity) {verbose = verbosity;}

    size_t getNumElements(unsigned int downsamp=1) {
      return dimTime(downsamp) * nchans_read_;  
    }

    size_t getDataSize(unsigned int downsamp=1) {return getNumElements(downsamp) * sizeof(float); }

    matrixView<float>& dataView() { return dataView_; }

    const matrixView<float>& dataView() const { return dataView_; }

    const char* getFilename() {return filename_;}

    ~Fits() {};
  
  private:
    const char* filename_;
    
    // Strings stored in char arrays (fixed-size buffers)
    char telescope_name_[40]  = {};  // TELESCOP
    char instrument_[100]     = {};  // BACKEND
    char object_name_[100]    = {};  // SRC_NAME
    char ra_[40]              = {};  // RA
    char dec_[40]             = {};  // DEC
    char observer_[100]       = {};  // OBSERVER
    char projid_[100]         = {};  // PROJID
    char dateobs_[100]        = {};  // DATE-OBS
    char ephem_[10]           = "DE405";
    char rastring_[50];
    char decstring_[50];
    char obs_[3];
    char outscope_[40];
    
    int         num_hdus_          = 0;
    int         stt_imjd_          = 0;     // STT_IMJD
    int         stt_smjd_          = 0;     // STT_SMJD
    double      stt_offs_          = 0.0;   // STT_OFFS
    double      epoch_             = 0.0;   // computed: IMJD + (SMJD + OFFS)/86400
    double      be_delay_          = 0.0;   // Backend Delay

    // SUBINT HDU keys
    double      tbin_              = 0.0;   // TBIN
    int         nchans_read_       = 0;     // NCHAN
    int         nsblk_             = 0;     // NSBLK
    int         naxis1_            = 0;     // NAXIS1
    int         nbin_              = 0;     // NBIN
    int         npol_              = 0;     // NPOL
    int         naxis2_            = 0;     // NAXIS2
    int         verbose            = 0;

    size_t      data_byte_width_   = 0;     // nbin * nchan * npol * nsblk (bytes if 1 byte/sample)
    size_t      scal_offs_width_   = 0;     // nchan * npol
    size_t fits_header_bytesize_   = 0;     // Offset for data in the binary table (bytes)
    size_t fits_data_bytesize_     = 0;     // Size of fits data block (bytes)
    size_t file_size_              = 0;     // File size in bytes
    size_t file_size_aligned_      = 0;     // File size aligned to 4096 bytes
    size_t timeseries_col_byte_offset_        = 0;     // Offset to reach timeseries data col (bytes)

    std::vector<double> freqs_;             // DAT_FREQ column (size = nchan)

    float* data_                            = nullptr; 
    unsigned char* aligned_filesize_buffer_ = nullptr;
    matrixView<float> dataView_;
};

#endif