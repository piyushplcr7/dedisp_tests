#include "fits.hpp"
#include <cstdint>
#include <iostream>
#include <vector>
#include "fitsio.h"
#include <string.h>
#include <chrono>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <tuple>
#include <string>
#include <fstream>
#include "matrix_view.hpp"
#include "barycenter_presto_utils.h"

Fits::Fits(const char* filename) {
  // assign the filename
  filename_ = filename;

  // Extract relevant info using the cfitsio library
  fitsfile *ffptr;
  int status = 0;
  fits_open_file(&ffptr, filename_, READONLY, &status);
  if (status != 0) {
    printf("Error in opening fits file %s\n", filename_);
    exit(1);
  }

  char comment[80];

  // Get number of HDUs
  fits_get_num_hdus(ffptr, &num_hdus_, &status);
  fits_read_key(ffptr, TSTRING, "TELESCOP", telescope_name_, comment, &status);
  fits_read_key(ffptr, TSTRING, "BACKEND", instrument_, comment, &status);
  fits_read_key(ffptr, TSTRING, "SRC_NAME", object_name_, comment, &status);
  fits_read_key(ffptr, TSTRING, "RA", ra_, comment, &status);
  fits_read_key(ffptr, TSTRING, "DEC", dec_, comment, &status);
  fits_read_key(ffptr, TSTRING, "OBSERVER", observer_, comment, &status);

  if (fits_read_key(ffptr, TINT, "STT_IMJD", &stt_imjd_, comment, &status)) {
    fits_report_error(stderr, status);
    exit(1);
  }

  if (fits_read_key(ffptr, TINT, "STT_SMJD", &stt_smjd_, comment, &status)) {
    fits_report_error(stderr, status);
    exit(1);
  }

  if (fits_read_key(ffptr, TDOUBLE, "STT_OFFS", &stt_offs_, comment, &status)) {
    fits_report_error(stderr, status);
    exit(1);
  }

  if (fits_read_key(ffptr, TDOUBLE, "BE_DELAY", &be_delay_, comment, &status)) {
    fits_report_error(stderr, status);
    exit(1);
  }

  if (fits_read_key(ffptr, TSTRING, "PROJID", projid_, comment, &status)) {
    fits_report_error(stderr, status);
    exit(1);
  }

  if (fits_read_key(ffptr, TSTRING, "DATE-OBS", dateobs_, comment, &status)) {
    fits_report_error(stderr, status);
    exit(1);
  }

  // Moving to the relevant hdu
  fits_movnam_hdu(ffptr, BINARY_TBL, "SUBINT", 0, &status);
  fits_read_key(ffptr, TDOUBLE, "TBIN", &tbin_, comment, &status);
  fits_read_key(ffptr, TINT, "NCHAN", &nchans_read_, comment, &status);
  fits_read_key(ffptr, TINT, "NSBLK", &nsblk_, comment, &status);
  fits_read_key(ffptr, TINT, "NAXIS1", &naxis1_, comment, &status);
  fits_read_key(ffptr, TINT, "NBIN", &nbin_, comment, &status);
  fits_read_key(ffptr, TINT, "NPOL", &npol_, comment, &status);
  fits_read_key(ffptr, TINT, "NAXIS2", &naxis2_, comment, &status);
    
  freqs_.assign(nchans_read_, 0.0);

  int colnum = -1;
  fits_get_colnum(ffptr, 0, "DAT_FREQ", &colnum, &status);

  if (status == COL_NOT_FOUND) {
    printf("Warning!:  Can't find the channel freq column!\n");
    status = 0; // Reset status
  } else {
    int anynull = 0;
    fits_read_col(ffptr, TDOUBLE, colnum, 1L, 1L, nchans_read_, 0, freqs_.data(),
                  &anynull, &status);
  }

  // Computing some quantities

  // Width of scales and offsets per subint
  scal_offs_width_ = (size_t)nchans_read_ * npol_;
  // Byte width of the time series series data column
  data_byte_width_ = (size_t)nbin_ * nchans_read_ * npol_ * nsblk_;

  // TloTOA
  epoch_ = stt_imjd_ + ((stt_smjd_ + stt_offs_) / 86400.0);

  telescope_to_tempocode(telescope_name_, outscope_, obs_);
  getTempoStrings(ra_, dec_, rastring_, decstring_);

  fits_close_file(ffptr, &status);

  // Finding byte size of HDUs
  FILE *fptr;

  if ((fptr = fopen(filename_, "rb")) == NULL) {
    printf("Error! opening file");
    exit(1);
  }

  int i = 0;
  int num_hdus = num_hdus_;

  char smallbuffer[80];
  for (; i < 1000; ++i) {
    if (num_hdus == 0) {
      break;
    }
    for (int j = 0; j < 36; ++j) {
      fread(smallbuffer, 1, 80, fptr);
      // Find END
      char *tempptr = strstr(smallbuffer, "END     ");
      if (tempptr != NULL) {
        num_hdus--;
      }
    }
  }
  fclose(fptr);

  // Finding important sizes and offsets
  fits_header_bytesize_ = (size_t)i * 2880;
  fits_data_bytesize_ = (size_t)naxis1_ * naxis2_;
  file_size_ = fits_header_bytesize_+ fits_data_bytesize_;
  file_size_aligned_ = ((file_size_ + 4095) / 4096) * 4096;
  timeseries_col_byte_offset_ =
    (size_t)naxis1_ - (size_t)4 * nchans_read_ - (size_t)2 * 4 * scal_offs_width_ - data_byte_width_;

}

void Fits::printInfo() const noexcept {
  printf("------------------------------ FITS INFO  "
         "----------------------------\n");
  printf("filename         = %s\n", filename_);
  printf("lo, hi freq      = %.15f, %.15f\n", freqs_[0], freqs_[nchans_read_ - 1]);
  printf("No. of HDUs      = %d\n", num_hdus_);
  printf("telescope name   = %s \n", telescope_name_);
  printf("instrument name  = %s \n", instrument_);
  printf("object name      = %s \n", object_name_);
  printf("right_ascension  = %s \n", ra_);
  printf("declination      = %s \n", dec_ + 1);
  printf("observer         = %s \n", observer_);
  printf("STT_IMJD         = %d  \n", stt_imjd_);
  printf("STT_SMJD         = %d  \n", stt_smjd_);
  printf("STT_OFFS         = %.15f \n", stt_offs_);
  printf("epoch/tlotoa     = %.15f\n", epoch_);
  printf("projid           = %s   \n", projid_);
  printf("dateobs          = %s   \n", dateobs_);
  printf("read tbin        = %f\n", tbin_);
  printf("Nchans read      = %d\n", nchans_read_);
  printf("nsblk            = %d\n", nsblk_);
  printf("naxis1           = %d\n", naxis1_);
  printf("nbin             = %d\n", nbin_);
  printf("npol             = %d\n", npol_);
  printf("data byte width  = %ld\n", data_byte_width_);
  printf("offs, scal width = %ld\n", scal_offs_width_);
  printf("naxis2           = %d\n", naxis2_);
  printf("obs              = %s\n", obs_);
  printf("rastring         = %s\n", rastring_);
  printf("decstring        = %s\n", decstring_);
}


void Fits::reduceData(unsigned char* outBuf, int out_bits, int poln, unsigned int downsamp) {
#ifdef TESTDEDISP_DEBUG
  auto start_time = std::chrono::high_resolution_clock::now();
#endif
  if (out_bits == 32) {
    float* data = reinterpret_cast<float*>(outBuf);

    if (downsamp == 1) {
      reduceBinaryTable(aligned_filesize_buffer_, data, poln, naxis1_, naxis2_,
                      nsblk_, nchans_read_, npol_, fits_header_bytesize_, timeseries_col_byte_offset_,
                      scal_offs_width_);
    }
    else if (downsamp > 1) {
      reduceBinaryTableDownSamp(aligned_filesize_buffer_, data, poln, naxis1_, naxis2_,
                      nsblk_, nchans_read_, npol_, fits_header_bytesize_, timeseries_col_byte_offset_,
                      scal_offs_width_, downsamp);

      size_t reducedDataLen = getNumElements(downsamp);
      #pragma omp parallel for schedule(static)
      for (size_t i = 0 ; i < reducedDataLen ; ++i) {
        data[i] /= downsamp;
      }
    }
  }
  else if (out_bits == 8) {
    unsigned char* data = outBuf;
    if (downsamp == 1) {
      reduceBinaryTable8(aligned_filesize_buffer_, data, poln, naxis1_, naxis2_,
                      nsblk_, nchans_read_, npol_, fits_header_bytesize_, timeseries_col_byte_offset_,
                      scal_offs_width_);
    }
    else if (downsamp > 1) {
      reduceBinaryTableDownSamp8(aligned_filesize_buffer_, data, poln, naxis1_, naxis2_,
                      nsblk_, nchans_read_, npol_, fits_header_bytesize_, timeseries_col_byte_offset_,
                      scal_offs_width_, downsamp);
      // Division with rounding is now done inside reduceBinaryTableDownSamp8
    }
  }
  else {
    std::cerr << "Unsupported out_bits value!" << std::endl;
    std::abort();
  }
  
  
#ifdef TESTDEDISP_DEBUG
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();

  std::cout << "reduction finished, time: " << (double)duration_us / 1e6
            << std::endl;
#endif
}


void Fits::reduceBinaryTable(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width) {
  // Skipping the header
  unsigned char *bin_table_start = full_binary_table + data_offset_from_start;
  size_t nchans_poln = nchans * poln;

// Going over all the rows (subints) of the binary table
#pragma omp parallel for schedule(static)
  for (size_t subint = 0; subint < naxis2; ++subint) 
  {
    // Position for data_wts for a subint
    float *data_wts =
        (float *)(bin_table_start + subint * naxis1 + data_cols_offset);
    // data wts has nchans floats, after which data_offs starts
    float *data_offs = data_wts + nchans;
    // data_offs has scal_offs_width floats, after which data_scl starts
    float *data_scl = data_offs + scal_offs_width;
    // data_scl has scal_offs_width floats, after which rawdata starts
    unsigned char *rawdata = (unsigned char *)(data_scl + scal_offs_width);

    size_t subIntStartTimeIdx = nsblk * subint;

    for (int chan = 0; chan < nchans; ++chan) 
    {
      // Byte swapping for 3 floats directly. Done once for a subint!
      swap_endian_3floats(data_scl[nchans_poln + chan],
                          data_offs[nchans_poln + chan], data_wts[chan]);
    }

    for (size_t spectra = 0; spectra < nsblk; ++spectra) 
    {
      size_t outTimeIdx = subIntStartTimeIdx + spectra;
      size_t nchans_npol_spectra_nchans_poln =
          nchans * npol * spectra + nchans_poln;
      for (size_t chan = 0; chan < nchans; ++chan) 
      {
        // No byteswapping
        // (nsblk * subint + spectra) * nchans
        data[outTimeIdx * nchans + chan] =
            ((float)rawdata[nchans_npol_spectra_nchans_poln + chan] *
                  data_scl[nchans_poln + chan] +
              data_offs[nchans_poln + chan]) *
            data_wts[chan];
      }
    }
  }
}

void Fits::reduceBinaryTableDownSamp(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width, int downsamp) {
  // Skipping the header
  unsigned char *bin_table_start = full_binary_table + data_offset_from_start;
  size_t nchans_poln = nchans * poln;

// Going over all the rows (subints) of the binary table
#pragma omp parallel for schedule(static)
  for (size_t subint = 0; subint < naxis2; ++subint) 
  {
    // Position for data_wts for a subint
    float *data_wts =
        (float *)(bin_table_start + subint * naxis1 + data_cols_offset);
    // data wts has nchans floats, after which data_offs starts
    float *data_offs = data_wts + nchans;
    // data_offs has scal_offs_width floats, after which data_scl starts
    float *data_scl = data_offs + scal_offs_width;
    // data_scl has scal_offs_width floats, after which rawdata starts
    unsigned char *rawdata = (unsigned char *)(data_scl + scal_offs_width);

    size_t subIntStartTimeIdx = nsblk * subint;

    for (int chan = 0; chan < nchans; ++chan) 
    {
      // Byte swapping for 3 floats directly. Done once for a subint!
      swap_endian_3floats(data_scl[nchans_poln + chan],
                          data_offs[nchans_poln + chan], data_wts[chan]);
    }

    for (size_t spectra = 0; spectra < nsblk; ++spectra) 
    {
      size_t outTimeIdx = (subIntStartTimeIdx + spectra)/downsamp;
      size_t nchans_npol_spectra_nchans_poln =
          nchans * npol * spectra + nchans_poln;
      for (size_t chan = 0; chan < nchans; ++chan) 
      {
        // No byteswapping
        // ((nsblk * subint + spectra)/downsamp) * nchans

        // Ensuring nsblk % downsamp == 0 means no clashes between threads over subints
        //#pragma omp atomic update
        data[outTimeIdx * nchans + chan] +=
            ((float)rawdata[nchans_npol_spectra_nchans_poln + chan] *
                  data_scl[nchans_poln + chan] +
              data_offs[nchans_poln + chan]) *
            data_wts[chan];
      }
    }
  }
}

void Fits::reduceBinaryTable8(unsigned char *full_binary_table, unsigned char *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width) {
  // Skipping the header
  unsigned char *bin_table_start = full_binary_table + data_offset_from_start;
  size_t nchans_poln = nchans * poln;

// Going over all the rows (subints) of the binary table
#pragma omp parallel for schedule(static)
  for (size_t subint = 0; subint < naxis2; ++subint) 
  {
    float *data_wts =
        (float *)(bin_table_start + subint * naxis1 + data_cols_offset);
    // data wts has nchans floats, after which data_offs starts
    float *data_offs = data_wts + nchans;
    // data_offs has scal_offs_width floats, after which data_scl starts
    float *data_scl = data_offs + scal_offs_width;
    // data_scl has scal_offs_width floats, after which rawdata starts
    unsigned char *rawdata = (unsigned char *)(data_scl + scal_offs_width);
    size_t subIntStartTimeIdx = nsblk * subint;

    for (size_t spectra = 0; spectra < nsblk; ++spectra) 
    {
      size_t outTimeIdx = subIntStartTimeIdx + spectra;
      size_t nchans_npol_spectra_nchans_poln =
          nchans * npol * spectra + nchans_poln;
      for (size_t chan = 0; chan < nchans; ++chan) 
      {
        // No byteswapping
        // (nsblk * subint + spectra) * nchans
        data[outTimeIdx * nchans + chan] = rawdata[nchans_npol_spectra_nchans_poln + chan] ;
      }
    }
  }
}

#ifdef TESTDEDISP_DEBUG
void Fits::writeModifiedFits(const char* outfile) const {
  // Read the entire original file into a buffer
  FILE* fin = fopen(filename_, "rb");
  if (!fin) {
    fprintf(stderr, "Fits::writeModifiedFits: cannot open '%s' for reading\n", filename_);
    return;
  }
  fseek(fin, 0, SEEK_END);
  size_t total_size = (size_t)ftell(fin);
  fseek(fin, 0, SEEK_SET);

  std::vector<unsigned char> buf(total_size);
  if (fread(buf.data(), 1, total_size, fin) != total_size) {
    fprintf(stderr, "Fits::writeModifiedFits: short read on '%s'\n", filename_);
    fclose(fin);
    return;
  }
  fclose(fin);

  // Compute the scale/offset/weight values that make the 32-bit reduction
  // formula produce ((float)val - 127.5f) / nchan:
  //   (val * scl + offs) * wt = (val - 127.5) / nchan
  //   => scl = 1.0/nchan, offs = -127.5/nchan, wt = 1.0
  float scl_val = 1.0f / (float)nchans_read_;
  float offs_val = -127.5f / (float)nchans_read_;
  float wts_val = 1.0f;

  // Convert to big-endian (FITS stores floats big-endian)
  auto to_big_endian = [](float f) -> uint32_t {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(float));
    return bswap_32(bits);
  };

  uint32_t scl_be  = to_big_endian(scl_val);
  uint32_t offs_be = to_big_endian(offs_val);
  uint32_t wts_be  = to_big_endian(wts_val);

  // Overwrite data_wts, data_offs, data_scl for each subint row
  unsigned char* bin_table_start = buf.data() + fits_header_bytesize_;
  size_t nchans_poln = (size_t)nchans_read_ * 0; // poln=0

  for (size_t subint = 0; subint < (size_t)naxis2_; ++subint) {
    float* data_wts =
        (float*)(bin_table_start + subint * naxis1_ + timeseries_col_byte_offset_);
    float* data_offs = data_wts + nchans_read_;
    float* data_scl  = data_offs + scal_offs_width_;

    // Overwrite weights (nchans floats)
    for (int c = 0; c < nchans_read_; ++c) {
      memcpy(&data_wts[c], &wts_be, sizeof(uint32_t));
    }

    // Overwrite offsets and scales for each polarization
    for (size_t p = 0; p < (size_t)npol_; ++p) {
      size_t idx = p * nchans_read_;
      for (int c = 0; c < nchans_read_; ++c) {
        memcpy(&data_offs[idx + c], &offs_be, sizeof(uint32_t));
        memcpy(&data_scl[idx + c], &scl_be, sizeof(uint32_t));
      }
    }
  }

  // Write the modified buffer to the output file
  FILE* fout = fopen(outfile, "wb");
  if (!fout) {
    fprintf(stderr, "Fits::writeModifiedFits: cannot open '%s' for writing\n", outfile);
    return;
  }
  fwrite(buf.data(), 1, total_size, fout);
  fclose(fout);

  printf("Fits::writeModifiedFits: wrote %s (%zu bytes)\n", outfile, total_size);
}
#endif // TESTDEDISP_DEBUG

void Fits::reduceBinaryTableDownSamp8(unsigned char *full_binary_table, unsigned char *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width, int downsamp) {
  // Skipping the header
  unsigned char *bin_table_start = full_binary_table + data_offset_from_start;
  size_t nchans_poln = nchans * poln;

  // Accumulate into uint16_t to avoid overflow and integer truncation
  size_t outLen = ((size_t)naxis2 * nsblk / downsamp) * nchans;
  std::vector<uint16_t> acc(outLen, 0);

// Going over all the rows (subints) of the binary table
#pragma omp parallel for schedule(static)
  for (size_t subint = 0; subint < naxis2; ++subint) 
  {
    float *data_wts =
        (float *)(bin_table_start + subint * naxis1 + data_cols_offset);
    // data wts has nchans floats, after which data_offs starts
    float *data_offs = data_wts + nchans;
    // data_offs has scal_offs_width floats, after which data_scl starts
    float *data_scl = data_offs + scal_offs_width;
    // data_scl has scal_offs_width floats, after which rawdata starts
    unsigned char *rawdata = (unsigned char *)(data_scl + scal_offs_width);
    size_t subIntStartTimeIdx = nsblk * subint;

    for (size_t spectra = 0; spectra < nsblk; ++spectra) 
    {
      size_t outTimeIdx = (subIntStartTimeIdx + spectra)/downsamp;
      size_t nchans_npol_spectra_nchans_poln =
          nchans * npol * spectra + nchans_poln;
      for (size_t chan = 0; chan < nchans; ++chan) 
      {
        // Ensuring nsblk % downsamp == 0 means no clashes between threads over subints
        acc[outTimeIdx * nchans + chan] += rawdata[nchans_npol_spectra_nchans_poln + chan];
      }
    }
  }

  // Round and write back to unsigned char
  #pragma omp parallel for schedule(static)
  for (size_t i = 0; i < outLen; ++i) {
    data[i] = (unsigned char)((acc[i] + downsamp / 2) / downsamp);
  }
}


