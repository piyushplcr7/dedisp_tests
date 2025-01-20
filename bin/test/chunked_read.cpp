
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <unistd.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include "fdd_gpu.h"
#include "fitsio.h"
#include <byteswap.h>
#include <cstring>

#include <Plan.hpp>
#include <cuda_runtime.h>

#include "fdd/helper.h"

static int rows = 200;
static long bytes_per_row = 123002956;
static long bytes_to_be_read = bytes_per_row * rows;
double megabytes_to_be_read = (double)bytes_to_be_read / 1e6;

void swapEndian(float *val) {
  unsigned char *valPtr = (unsigned char *)val;
  unsigned char temp;

  // Swap bytes in place
  temp = valPtr[0];
  valPtr[0] = valPtr[3];
  valPtr[3] = temp;

  temp = valPtr[1];
  valPtr[1] = valPtr[2];
  valPtr[2] = temp;
}

inline float swap_endian_float(float val) {
  uint32_t int_rep;
  std::memcpy(&int_rep, &val, sizeof(val));
  int_rep = (int_rep >> 24) | ((int_rep << 8) & 0x00FF0000) |
            ((int_rep >> 8) & 0x0000FF00) | (int_rep << 24);
  float result;
  std::memcpy(&result, &int_rep, sizeof(result));
  return result;
}

inline void swap_endian_3floats(float &f1, float &f2, float &f3) {
  uint32_t *p1 = reinterpret_cast<uint32_t *>(&f1);
  uint32_t *p2 = reinterpret_cast<uint32_t *>(&f2);
  uint32_t *p3 = reinterpret_cast<uint32_t *>(&f3);

  *p1 = bswap_32(*p1);
  *p2 = bswap_32(*p2);
  *p3 = bswap_32(*p3);
}

void reduceBinaryTable(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width) {
  // Skipping the header
  unsigned char *bin_table_start = full_binary_table + data_offset_from_start;
  size_t nchans_poln = nchans * poln;

// Going over all the rows (subints) of the binary table
#pragma omp parallel
  {
    int num_threads = omp_get_num_threads();
    // std::cout << "omp_num_threads = " << num_threads << std::endl;
    int thread_id = omp_get_thread_num();

#pragma omp for
    for (size_t subint = 0; subint < naxis2; ++subint) {

      // Position for data_wts for a subint
      float *data_wts =
          (float *)(bin_table_start + subint * naxis1 + data_cols_offset);
      // data wts has nchans floats, after which data_offs starts
      float *data_offs = data_wts + nchans;
      // data_offs has scal_offs_width floats, after which data_scl starts
      float *data_scl = data_offs + scal_offs_width;
      // data_scl has scal_offs_width floats, after which rawdata starts
      unsigned char *rawdata = (unsigned char *)(data_scl + scal_offs_width);

      size_t nsblk_nchans_subint = nsblk * nchans * subint;

      for (int chan = 0; chan < nchans; ++chan) {
        // Byte swapping for 3 floats directly. Done once for a subint!
        swap_endian_3floats(data_scl[nchans_poln + chan],
                            data_offs[nchans_poln + chan], data_wts[chan]);
      }

      for (size_t spectra = 0; spectra < nsblk; ++spectra) {
        // size_t nchans_spectra = nchans * spectra;
        size_t nsblk_nchans_subint_nchans_spectra =
            nsblk_nchans_subint + nchans * spectra;
        // size_t nchans_npol_spectra = nchans * npol * spectra;
        size_t nchans_npol_spectra_nchans_poln =
            nchans * npol * spectra + nchans_poln;
        for (size_t chan = 0; chan < nchans; ++chan) {
          /* data[nsblk * nchans * subint + nchans * spectra + chan] =
            ((float)rawdata[nchans*npol*spectra + nchans*poln+chan]*
              data_scl[nchans*poln+chan]
              +data_offs[nchans*poln+chan])
              *data_wts[chan]; */

          // No byteswapping
          data[nsblk_nchans_subint_nchans_spectra + chan] =
              ((float)rawdata[nchans_npol_spectra_nchans_poln + chan] *
                   data_scl[nchans_poln + chan] +
               data_offs[nchans_poln + chan]) *
              data_wts[chan];

          // float swap_endian_float(float)
          /* data[nsblk_nchans_subint_nchans_spectra + chan] =
              ((float)rawdata[nchans_npol_spectra_nchans_poln + chan] *
                   swap_endian_float(data_scl[nchans_poln + chan]) +
               swap_endian_float(data_offs[nchans_poln + chan])) *
              swap_endian_float(data_wts[chan]); */

          /* data[nsblk_nchans_subint + nchans_spectra + chan] =
            ((float)rawdata[nchans_npol_spectra + nchans_poln+chan]*
              data_scl[nchans_poln+chan]
              +data_offs[nchans_poln+chan])
              *data_wts[chan]; */
        }
      }
    }
  }
}

long getDataFromRows(int fd, unsigned char *table_data, long chunksize,
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

int main(int argc, char **argv) {
  const char *filename = "/home/pp/mwa_data/test.fits";

  // Using CFITSIO routines to get important parameters
  fitsfile *ffptr;
  int status = 0;
  fits_open_file(&ffptr, filename, READONLY, &status);
  if (status != 0) {
    printf("Error in opening fits file!\n");
    exit(1);
  }

  // Get number of HDUs
  int num_hdus;
  fits_get_num_hdus(ffptr, &num_hdus, &status);
  printf("No. of HDUs = %d\n", num_hdus);

  // Moving to the relevant hdu to get the data dimensions
  char comment[80];
  fits_movnam_hdu(ffptr, BINARY_TBL, "SUBINT", 0, &status);

  int naxis1;
  fits_read_key(ffptr, TINT, "NAXIS1", &naxis1, comment, &status);
  printf("naxis1 = %d\n", naxis1);

  int naxis2;
  fits_read_key(ffptr, TINT, "NAXIS2", &naxis2, comment, &status);
  printf("naxis2 = %d\n", naxis2);

  int nchans;
  fits_read_key(ffptr, TINT, "NCHAN", &nchans, comment, &status);
  printf("Nchans read = %d\n", nchans);

  int nsblk;
  fits_read_key(ffptr, TINT, "NSBLK", &nsblk, comment, &status);
  printf("nsblk = %d\n", nsblk);

  int npol;
  fits_read_key(ffptr, TINT, "NPOL", &npol, comment, &status);
  printf("npol = %d\n", npol);

  int nbin;
  fits_read_key(ffptr, TINT, "NBIN", &nbin, comment, &status);
  printf("nbin = %d\n", nbin);

  fits_close_file(ffptr, &status);

  // Reading the fits file manually to find the byte size of HDUs, ie offset
  // after which data begins
  FILE *fptr;

  if ((fptr = fopen(filename, "rb")) == NULL) {
    printf("Error! opening file");
    exit(1);
  }

  int i = 0;

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

  size_t offset_for_data = i * 2880;
  size_t data_size = (size_t)naxis1 * naxis2;
  std::cout << "i = " << i << std::endl;
  std::cout << "offset_for_data = " << offset_for_data << std::endl;
  std::cout << "naxis1 * naxis2 = " << (size_t)naxis1 * naxis2 << std::endl;
  std::cout << "data size = " << data_size << std::endl;
  size_t file_size = offset_for_data + data_size;
  // size_t file_size_aligned = (file_size/4096+1)*4096;
  size_t file_size_aligned = ((file_size + 4095) / 4096) * 4096;
  std::cout << "file size = " << file_size
            << ", aligned size = " << file_size_aligned << std::endl;

  size_t scal_offs_width = (size_t)nchans * npol;
  size_t data_byte_width = (size_t)nbin * nchans * npol * nsblk;
  size_t initial_offset =
      (size_t)naxis1 - 4 * nchans - 2 * 4 * scal_offs_width - data_byte_width;

  // Longest chunksize 2147479552 (linux read documentation)
  long chunksize = 2147479552;

  int fd = open(filename, O_RDONLY | O_DIRECT);
  int fd_nodirect = open(filename, O_RDONLY);

  if (fd == -1) {
    perror("open");
    return -1;
  }

  unsigned char *table_full;
  if (posix_memalign((void **)&table_full, 4096, file_size_aligned) != 0) {
    perror("posix_memalign");
    close(fd);
    return -1;
  }

  auto start_time = std::chrono::high_resolution_clock::now();

  long chunks =
      getDataFromRows(fd, table_full, chunksize, file_size, fd_nodirect);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();
  // long megabytes_read = (double) (chunks * chunksize )/1e6;
  long megabytes_read = (double)(file_size) / 1e6;
  std::cout << "read " << chunks << " chunks with chunksize = " << chunksize
            << ", time: " << (double)duration_us / 1e6 << " seconds"
            << ", Read speed (MB/s): "
            << megabytes_read / (double)duration_us * 1e6 << std::endl;

  close(fd);

  float *rawdata;
  rawdata = (float *)calloc((size_t)nsblk * naxis2 * nchans, sizeof(float));
  int poln = 0;

  start_time = std::chrono::high_resolution_clock::now();
  reduceBinaryTable(table_full, rawdata, poln, naxis1, naxis2, nsblk, nchans,
                    npol, offset_for_data, initial_offset, scal_offs_width);

  end_time = std::chrono::high_resolution_clock::now();
  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();

  std::cout << "reduction finished, time: " << (double)duration_us / 1e6
            << std::endl;

  free(table_full);
  free(rawdata);

  /* std::cout << "##############################" << std::endl;
  std::cout << "Using mmap" << std::endl;

  int fd1 = open(filename, O_RDONLY);
  if (fd1 == -1) {
    perror("Error opening file");
    return EXIT_FAILURE;
  }

  struct stat file_stat;
  if (fstat(fd1, &file_stat) == -1) {
    perror("Error getting file size");
    close(fd1);
    return EXIT_FAILURE;
  }

  size_t file_size_stat = file_stat.st_size;
  void *mapped = mmap(NULL, file_size_stat, PROT_READ, MAP_PRIVATE, fd1, 0);
  if (mapped == MAP_FAILED) {
    perror("Error mapping file");
    close(fd1);
    return EXIT_FAILURE;
  }

  float *rawdata1;
  rawdata1 = (float *)calloc((size_t)nsblk * naxis2 * nchans, sizeof(float));
  unsigned char *table_full1 = (unsigned char *)mapped;

  start_time = std::chrono::high_resolution_clock::now();
  reduceBinaryTable(table_full1, rawdata1, poln, naxis1, naxis2, nsblk, nchans,
                    npol, offset_for_data, initial_offset, scal_offs_width);
  end_time = std::chrono::high_resolution_clock::now();
  duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();

  std::cout << "MMAP+reduction finished, time: " << (double)duration_us / 1e6
            << std::endl; */
  /* std::cout << "Comparing two tables" << std::endl;
  size_t reducedDataSize = (size_t)nsblk * naxis2 * nchans;
  for (size_t i = 0; i < reducedDataSize; ++i) {
    if (rawdata[i] != rawdata1[i]) {
      std::cout << "Mismatch at i = " << i << std::endl;
    }
  }
  std::cout << "Perfect Match!!" << std::endl; */
  //free(rawdata1);

  return 0;
}