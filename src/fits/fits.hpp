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
    size_t      dimTime()      const { return naxis2_ * nsblk_; }
    size_t      fileSizeAligned() const noexcept { return file_size_aligned_; }
    size_t      fileSize()     const { return file_size_; }
    float*      data()         const noexcept { return data_; }
    int         naxis1()       const noexcept {return naxis1_; }

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
      return ((size_t)nchans_read_ * nsblk_ * naxis2_)/downsamp; 
    }

    size_t getDataSize(unsigned int downsamp=1) {return getNumElements(downsamp) * sizeof(float); }

    matrixView<float>& dataView() { return dataView_; }

    const matrixView<float>& dataView() const { return dataView_; }

    ~Fits() {};
  
  private:
    const char* filename_;
    
    // Strings stored in char arrays (fixed-size buffers)
    char telescope_name_[40] = {};  // TELESCOP
    char instrument_[100]     = {};  // BACKEND
    char object_name_[100]    = {};  // SRC_NAME
    char ra_[40]             = {};  // RA
    char dec_[40]            = {};  // DEC
    char observer_[100]       = {};  // OBSERVER
    char projid_[100]         = {};  // PROJID
    char dateobs_[100]        = {};  // DATE-OBS
    constexpr static char ephem[10]     = "DE405";
    
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