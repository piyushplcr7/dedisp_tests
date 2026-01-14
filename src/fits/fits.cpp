#include "fits.hpp"
#include <iostream>
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

  if (verbose) {
    printf("lo, hi freq = %.15f, %.15f\n", freqs_[0], freqs_[nchans_read_ - 1]);
    printf("No. of HDUs = %d\n", num_hdus_);
    printf("telescope name = %s \n", telescope_name_);
    printf("instrument name = %s \n", instrument_);
    printf("object name = %s \n", object_name_);
    printf("right_ascension = %s \n", ra_);
    printf("declination = %s \n", dec_ + 1);
    printf("observer = %s \n", observer_);
    printf("STT_IMJD = %d   (%s)\n", stt_imjd_, comment);
    printf("STT_SMJD = %d   (%s)\n", stt_smjd_, comment);
    printf("STT_OFFS = %.15f   (%s)\n", stt_offs_, comment);
    printf("Computed epoch = %.15f\n", epoch_);
    printf("projid = %s   (%s)\n", projid_, comment);
    printf("dateobs = %s   (%s)\n", dateobs_, comment);
    printf("read tbin = %f\n", tbin_);
    printf("Nchans read = %d\n", nchans_read_);
    printf("nsblk = %d\n", nsblk_);
    printf("naxis1 = %d\n", naxis1_);
    printf("nbin = %d\n", nbin_);
    printf("npol = %d\n", npol_);
    printf("data byte width = %ld\n", data_byte_width_);
    printf("offs, scal width = %ld\n", scal_offs_width_);
    printf("naxis2 = %d\n", naxis2_);
  } 

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

void Fits::extractDataDirect(size_t chunksize) {
  if (aligned_filesize_buffer_ == nullptr) {
    std::cerr << "Error: aligned_filesize_buffer_ is not set. Use setAlignedFileSizeBuffer() before calling extractDataDirect()." << std::endl;
    exit(-1);
  }

  // Open the file using two file descriptors - one for direct read and one for normal read
  int fd = open(filename_, O_RDONLY | O_DIRECT);
  if (fd == -1) {
    perror("open");
    exit(-1);
  }

  int fd_nodirect = open(filename_, O_RDONLY);
  if (fd_nodirect == -1) {
    perror("open");
    close(fd);
    exit(-1);
  }

#ifdef DEDISP_BENCHMARK
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  long chunks =
      getDataFromRows(fd, aligned_filesize_buffer_, chunksize, file_size_, fd_nodirect);

#ifdef DEDISP_BENCHMARK
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();
    long megabytes_read = (double)(file_size_) / 1e6;

    if (verbose) {
        std::cout << "read " << chunks << " chunks with chunksize = " << chunksize
                << " from file " << std::string(filename_)
                << ", time: " << (double)duration_us / 1e6 << " seconds"
                << ", Read speed (MB/s): "
                << megabytes_read / (double)duration_us * 1e6 << std::endl;
    }
#endif


  close(fd);
  close(fd_nodirect);
}

void Fits::reduceData(int poln, unsigned int downsamp) {
  if (aligned_filesize_buffer_ == nullptr) {
    std::cerr << "Error: aligned_filesize_buffer_ is not set. Use setAlignedFileSizeBuffer() before calling reduceData()." << std::endl;
    exit(-1);
  }
  if (data_ == nullptr) {
    std::cerr << "Error: data_ is not set. Use setDataBuffer() before calling reduceData()." << std::endl;
    exit(-1);
  }

#ifdef DEDISP_BENCHMARK
  // Reducing the binary table to get the relevant data
  auto start_time = std::chrono::high_resolution_clock::now();
#endif

  if (downsamp == 1) {
    reduceBinaryTable(aligned_filesize_buffer_, data_, poln, naxis1_, naxis2_,
                    nsblk_, nchans_read_, npol_, fits_header_bytesize_, timeseries_col_byte_offset_,
                    scal_offs_width_);
  }
  else if (downsamp > 1) {
    reduceBinaryTableDownSamp(aligned_filesize_buffer_, data_, poln, naxis1_, naxis2_,
                    nsblk_, nchans_read_, npol_, fits_header_bytesize_, timeseries_col_byte_offset_,
                    scal_offs_width_, downsamp);
  }

  // Create the data view for easy access
  std::cout << "creating data view with " << (size_t)nsblk_ * naxis2_/downsamp << " rows and " << (size_t) nchans_read_ << " cols" << std::endl; 
  dataView_ = matrixView<float> (data_, (size_t)nsblk_ * naxis2_/downsamp , (size_t)nchans_read_);
  
#ifdef DEDISP_BENCHMARK
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();

  std::cout << "reduction finished, time: " << (double)duration_us / 1e6
            << std::endl;
#endif
}

long Fits::getDataFromRows(int fd, unsigned char *table_data, long chunksize,
                     long bytes_to_read, int fd_nodirect) {
  unsigned char *curr_pos;
  long chunk = 0;
  for (; chunk < bytes_to_read / chunksize; ++chunk) {
    curr_pos = table_data + chunksize * chunk;
    ssize_t bytes_read = read(fd, curr_pos, chunksize);
    if (bytes_read == -1) {
      perror("read");
      close(fd);
      exit(-1);
    }

    if (bytes_read == 0) {
      std::cerr << "Reached end of file prematurely" << std::endl;
      break;
    }

    if (bytes_read != chunksize) {
      std::cout << "read less than the chunksize: " << bytes_read << std::endl;
      // break;
    }
  }

  // Read last part of data using the alternative file descriptor if needed
  if (chunksize * chunk < bytes_to_read) {
    curr_pos = table_data + chunksize * chunk;
    ssize_t bytes_read =
        pread(fd_nodirect, curr_pos, bytes_to_read - chunksize * chunk,
              chunksize * chunk);

    if (bytes_read == -1) {
      perror("pread");
      close(fd);
      close(fd_nodirect);
      exit(-1);
    }

    if (bytes_read == 0) {
      std::cerr << "Reached end of file prematurely while reading last part"
                << std::endl;
    }
  }
  return chunk;
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

  // Check
  if (nsblk % downsamp != 0) {
    std::cerr << "Downsampling factor not allowed for reduction, choose something divisible by nsblk "
     << downsamp << "! Please choose a different value" << std::endl; 
     exit(-1);
  }

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


