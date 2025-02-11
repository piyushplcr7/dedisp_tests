/*
  Simple test application for libdedisp
  By Paul Ray (2013)
  With extended run method to use multiple different implementations
  (Dedisp PlanType) of dedispersion. (2020 ASTRON)
*/

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <byteswap.h>

#include <sys/mman.h>
#include <sys/stat.h>

#include "fdd_gpu.h"
#include "fitsio.h"
#include <byteswap.h>
#include <cstring>

#include <Plan.hpp>
#include <cuda_runtime.h>

#include "fdd/helper.h"

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

// Debug options
#define WRITE_INPUT_DATA 0
#define WRITE_OUTPUT_DATA 0

dedisp_float maxval_data = std::numeric_limits<float>::lowest();
dedisp_float minval_data = std::numeric_limits<float>::max();

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

void getDataFromRow(FILE *fptr, unsigned char *rawdata, float *data_scl,
                    float *data_offs, float *data_wts, int subint, 
                    size_t data_byte_width, size_t scal_offs_width, size_t nchans, size_t initial_offset) {
  // Move to dat_wts col
  fseek(fptr, initial_offset, SEEK_CUR); 

  // read dat_wts
  fread(data_wts, 4, nchans, fptr);
  // read data_offs
  fread(data_offs, 4, scal_offs_width, fptr);
  // read data_scl
  fread(data_scl, 4, scal_offs_width, fptr);
  // read data
  fread(rawdata, 1, data_byte_width, fptr);

  for (int i = 0; i < scal_offs_width; ++i) {
    swapEndian(&data_offs[i]);
    swapEndian(&data_scl[i]);
  }

  for (int i = 0; i < nchans; ++i) {
    swapEndian(&data_wts[i]);
  }
}

void reduceData(float *reduceddata, unsigned char *rawdata, float *data_scl,
                float *data_offs, float *data_wts, int poln, int subint,
                size_t nsblk, size_t nchans, int npol) {
  for (int spectra = 0; spectra < nsblk; ++spectra) {
    for (int chan = 0; chan < nchans; ++chan) {
      // Hard coded for zero_off = 0
      reduceddata[nsblk * nchans * subint + nchans * spectra + chan] =
          ((float)rawdata[nchans * npol * spectra + nchans * poln + chan] *
               data_scl[nchans * poln + chan] +
           data_offs[nchans * poln + chan]) *
          data_wts[chan];

      minval_data = std::min(
          reduceddata[nsblk * nchans * subint + nchans * spectra + chan],
          minval_data);
      maxval_data = std::max(
          reduceddata[nsblk * nchans * subint + nchans * spectra + chan],
          maxval_data);
    }
  }
}

void reduceDataTransposed(float *reduceddata, unsigned char *rawdata, float *data_scl,
                float *data_offs, float *data_wts, int poln, int subint,
                size_t nsblk, size_t nchans, int npol, int naxis2) {
  for (int spectra = 0; spectra < nsblk; ++spectra) {
    for (int chan = 0; chan < nchans; ++chan) {
      // Hard coded for zero_off = 0
      //reduceddata[nsblk * nchans * subint + nchans * spectra + chan] =
      reduceddata[chan * naxis2 * nsblk + subint * nsblk + spectra] =
          ((float)rawdata[nchans * npol * spectra + nchans * poln + chan] *
               data_scl[nchans * poln + chan] +
           data_offs[nchans * poln + chan]) *
          data_wts[chan];

      minval_data = std::min(
          reduceddata[nsblk * nchans * subint + nchans * spectra + chan],
          minval_data);
      maxval_data = std::max(
          reduceddata[nsblk * nchans * subint + nchans * spectra + chan],
          maxval_data);
    }
  }
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

#define READFROMFILE
#define WRITEFILES

int run(int argc, char **argv) {
  /////////////////////////////////////////////////////////////
  /////////// Parsing the command line arguments //////////////
  /////////////////////////////////////////////////////////////

  Cmdline *cmd = parseCmdline(argc, argv);
  showOptionValues();

  const char *filename = *cmd->argv;

  if (cmd->numdms >0) {
    if (cmd->dmstep == 0) {
      std::cerr << "ERROR: Non zero dmstep value required when specifying numdms!" << std::endl;
      exit(1);
    }
  }

  ////////////////////////////////////////////////////////////
  //// Extracting relevant parameters from the fits file /////
  ////////////////////////////////////////////////////////////

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
  
  char comment[80];
  // Moving to the relevant hdu
  fits_movnam_hdu(ffptr, BINARY_TBL, "SUBINT", 0, &status);

  double tbin;
  fits_read_key(ffptr, TDOUBLE, "TBIN", &tbin, comment, &status);
  printf("read tbin = %f\n", tbin);

  int nchans_read;
  fits_read_key(ffptr, TINT, "NCHAN", &nchans_read, comment, &status);
  printf("Nchans read = %d\n", nchans_read);

  int nsblk;
  fits_read_key(ffptr, TINT, "NSBLK", &nsblk, comment, &status);
  printf("nsblk = %d\n", nsblk);

  int naxis1;
  fits_read_key(ffptr, TINT, "NAXIS1", &naxis1, comment, &status);
  printf("naxis1 = %d\n", naxis1);

  int nbin;
  fits_read_key(ffptr, TINT, "NBIN", &nbin, comment, &status);
  printf("nbin = %d\n", nbin);

  int npol;
  fits_read_key(ffptr, TINT, "NPOL", &npol, comment, &status);
  printf("npol = %d\n", npol);

  size_t data_byte_width = (size_t)nbin * nchans_read * npol * nsblk;
  printf("data byte width = %ld\n", data_byte_width);
  size_t scal_offs_width = (size_t)nchans_read * npol;
  printf("offs, scal width = %ld\n",scal_offs_width);

  int naxis2;
  fits_read_key(ffptr, TINT, "NAXIS2", &naxis2, comment, &status);
  printf("naxis2 = %d\n", naxis2);

  int colnum = -1;
  fits_get_colnum(ffptr, 0, "DAT_FREQ", &colnum, &status);

  double *freqs = (double *) malloc(sizeof(double) * nchans_read);
  if (status == COL_NOT_FOUND) {
    printf("Warning!:  Can't find the channel freq column!\n");
    status = 0;     // Reset status
  } else {
    int anynull = 0;
    fits_read_col(ffptr, TDOUBLE, colnum, 1L, 1L,
                              nchans_read, 0, freqs, &anynull, &status);
  }

  printf("lo, hi freq = %.12f, %.12f\n", freqs[0], freqs[nchans_read-1]);

  fits_close_file(ffptr, &status);

  ///////////////////////////////////////////////////////////////////
  //////////////// Initializing more parameters /////////////////////
  ///////////////////////////////////////////////////////////////////

  dedisp_float dm_start = cmd->lodm;    // pc cm^-3
  dedisp_float dm_end = cmd->hidm;     // pc cm^-3
  dedisp_float pulse_width = cmd->pwidth; // ms
  dedisp_float downsamp = cmd->downsamp;

  int device_idx = 0;

  dedisp_float sampletime_base = tbin; 
  printf("sampletime base = %f\n",sampletime_base);
  
  dedisp_float Tobs = tbin * nsblk * naxis2;  // Observation duration in seconds
  printf("Tobs = %f\n",Tobs);

  dedisp_float dt = downsamp * sampletime_base; // s 
  printf("dt = %f\n",dt);

  dedisp_float f0 = freqs[nchans_read-1]; // MHz (highest channel!)
  printf("f0 = %f\n",f0);

  dedisp_size nchans = nchans_read;

  // (hifreq-lofreq)/(nchans-1)
  dedisp_float df = -1.0 * ((double)freqs[nchans_read-1] - (double)freqs[0]) / (nchans - 1); // MHz   (This must be negative!)
  printf("df = %.15f\n",df);

  dedisp_float bw = ((double)freqs[nchans_read-1] - (double)freqs[0]) / (nchans - 1) * nchans;  // MHz
  printf("bw = %f\n",bw);

  dedisp_size nsamps = Tobs / dt;
  dedisp_float dm_tol = 1.25;
  dedisp_size in_nbits = 8;
  dedisp_size out_nbits =
      32; // DON'T CHANGE THIS FROM 32, since that signals it to use floats

  dedisp_size dm_count;
  dedisp_size max_delay;
  dedisp_size nsamps_computed;
  dedisp_byte *input = 0;
  dedisp_float *output = 0;

  const dedisp_float *dmlist;

  clock_t startclock;

  /////////////////////////////////////////////////////////////
  //////////////// Finding byte size of HDUs //////////////////
  /////////////////////////////////////////////////////////////

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

  ////////////////////////////////////////////////////////////
  ///////////// Finding important sizes and offsets //////////
  ////////////////////////////////////////////////////////////

  size_t offset_for_data = i * 2880;
  size_t data_size = (size_t)naxis1 * naxis2;
  size_t file_size = offset_for_data + data_size;
  size_t file_size_aligned = ((file_size + 4095) / 4096) * 4096;
  size_t initial_offset =
      (size_t)naxis1 - 4 * nchans - 2 * 4 * scal_offs_width - data_byte_width;

  ////////////////////////////////////////////////////////////
  ///////////// Reading the whole file ///////////////////////
  ////////////////////////////////////////////////////////////

  // Longest chunksize 2147479552 for direct read (linux read documentation)
  long chunksize = 2147479552;

  // Opening file DIRECT and no DIRECT
  int fd = open(filename, O_RDONLY | O_DIRECT);
  int fd_nodirect = open(filename, O_RDONLY);

  if (fd == -1) {
    perror("open");
    return -1;
  }

  // Creating buffer to hold all file contents
  unsigned char *table_full;
  if (posix_memalign((void **)&table_full, 4096, file_size_aligned) != 0) {
    perror("posix_memalign");
    close(fd);
    return -1;
  }

  // Reading the whole file and timing the read
  auto start_time = std::chrono::high_resolution_clock::now();
  long chunks =
      getDataFromRows(fd, table_full, chunksize, file_size, fd_nodirect);
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();

  long megabytes_read = (double)(file_size) / 1e6;
  std::cout << "read " << chunks << " chunks with chunksize = " << chunksize
            << ", time: " << (double)duration_us / 1e6 << " seconds"
            << ", Read speed (MB/s): "
            << megabytes_read / (double)duration_us * 1e6 << std::endl;

  close(fd);
  close(fd_nodirect);

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

  std::cout << "summing rawdata channels into one" << std::endl;
  // Reducing rawdata to one timeseries by summing channels
  float *rawdata_summed;
  rawdata_summed = (float*) calloc((size_t) nsblk * naxis2, sizeof(float));

  for (size_t i = 0 ; i < (size_t) nsblk * naxis2 ; ++i) {
    for (int chan = 0 ; chan < nchans ; ++chan) {
        rawdata_summed[i] += rawdata[i*nchans+chan];
    }
  }

  std::cout << "summation finished" << std::endl;
  
  // Read the output binary file with const shift data
  FILE *fp_output = fopen("output.bin", "rb");
  if (!fp_output) {
    perror("Error opening files");
    exit(EXIT_FAILURE);
}
  
  nsamps_computed = 1992911;
  dm_count = 1595;
  size_t num_output = nsamps_computed * dm_count;
  float *data_output = (float *)malloc(num_output * sizeof(float));
  max_delay = 7089;
  size_t delay = 100; // the forced delay parameter in fdd_kernel.cuh. Shifts to the left!!

  size_t read1 = fread(data_output, sizeof(float), num_output, fp_output);

  
  for (int i = 0 ; i < 10 ; ++i) {
    std::cout << "summed rawdata, output.bin at i = " << i << " = [" << rawdata_summed[i+delay] << ", " << data_output[i] << "]" << std::endl;
  }

 size_t counts = 0;
  for (size_t i = 0; i < nsamps_computed ; ++i) {
    //float minval = std::min(fabs(data_output[i]), rawdata_summed[i+delay]);
    float maxval = std::max(fabs(data_output[i]), rawdata_summed[i+delay]);
    if (maxval < 1e-5){
      continue;
    } 
    if (fabs(data_output[i] - rawdata_summed[i+delay])/maxval/nchans > 1e-2) {
        std::cout << "mismatch at i = " << i << std::endl;
        std::cout << "summed rawdata, output.bin at i = " << i << " = [" << rawdata_summed[i+delay] << ", " << data_output[i] << "]" << std::endl;
        std::cout << fabs(data_output[i] - rawdata_summed[i+delay])/maxval/nchans << std::endl;
        //return -1;
    }
    counts++;
  }

  std::cout << "i at the end = " << counts << std::endl;
  
  #ifdef READFROMFILE
  free(rawdata);
  #endif
  printf("Dedispersion successful.\n");
  return 0;
}