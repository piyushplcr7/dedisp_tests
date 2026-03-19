#ifndef FITSHPP
#define FITSHPP

#include<vector>
#include<string>
#include <byteswap.h>
#include <cstdint>
#include <string>
#include <iostream>
#include "matrix_view.hpp"
#include "datafile.hpp"

inline void swap_endian_3floats(float &f1, float &f2, float &f3) {
  uint32_t *p1 = reinterpret_cast<uint32_t *>(&f1);
  uint32_t *p2 = reinterpret_cast<uint32_t *>(&f2);
  uint32_t *p3 = reinterpret_cast<uint32_t *>(&f3);

  *p1 = bswap_32(*p1);
  *p2 = bswap_32(*p2);
  *p3 = bswap_32(*p3);
}

class Fits : public dataFile {
  public:
    // Constructor with the filename
    Fits(const char* filename);

    // ----- Getters (read-only API) -----
    size_t      dimTime(unsigned int downsamp=1) const override { return ((size_t)naxis2_ * nsblk_)/downsamp; }
    int         naxis1()       const noexcept {return naxis1_; }
    //int         nsblk()        const noexcept {return nsblk_; }
    int         Tobs()         const noexcept {return tbin_ * (nsblk_ * naxis2_); }
 
    void        printInfo() const noexcept override;

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

    void reduceBinaryTable8(unsigned char *full_binary_table, unsigned char *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width);

    void reduceBinaryTableDownSamp8(unsigned char *full_binary_table, unsigned char *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width, int downsamp);

    void reduceData(unsigned char* outBuf, int out_bits=32, int poln=0, unsigned int downsamp=1) override;
    bool checkDownsamp(unsigned int downsamp) const override { return nsblk_ % downsamp == 0; }

    size_t getDataSize(unsigned int downsamp=1) const { return getNumElements(downsamp) * sizeof(float); }

    ~Fits() {};
  
  private:
    
    int         num_hdus_          = 0;
    int         stt_imjd_          = 0;     // STT_IMJD
    int         stt_smjd_          = 0;     // STT_SMJD
    double      stt_offs_          = 0.0;   // STT_OFFS
    double      be_delay_          = 0.0;   // Backend Delay

    // SUBINT HDU keys
    int         nsblk_   = 0;     // NSBLK
    int         naxis1_  = 0;     // NAXIS1
    int         naxis2_  = 0;     // NAXIS2

    size_t      data_byte_width_   = 0;     // nbin * nchan * npol * nsblk (bytes if 1 byte/sample)
    size_t      scal_offs_width_   = 0;     // nchan * npol
    size_t fits_header_bytesize_   = 0;     // Offset for data in the binary table (bytes)
    size_t fits_data_bytesize_     = 0;     // Size of fits data block (bytes)
    size_t timeseries_col_byte_offset_        = 0;     // Offset to reach timeseries data col (bytes)

};

#endif