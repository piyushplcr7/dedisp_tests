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
#include <string.h>
#include <time.h>

#include <byteswap.h>
#include <chrono>
#include <ctime>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

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
#include <vector>
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
#include "FDDGPUPlan.hpp"
#include "cufft_optimal_size.hpp"

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
                    size_t data_byte_width, size_t scal_offs_width,
                    size_t nchans, size_t initial_offset) {
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

void reduceDataTransposed(float *reduceddata, unsigned char *rawdata,
                          float *data_scl, float *data_offs, float *data_wts,
                          int poln, int subint, size_t nsblk, size_t nchans,
                          int npol, int naxis2) {
  for (int spectra = 0; spectra < nsblk; ++spectra) {
    for (int chan = 0; chan < nchans; ++chan) {
      // Hard coded for zero_off = 0
      // reduceddata[nsblk * nchans * subint + nchans * spectra + chan] =
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

template <typename PlanType> int run(int argc, char **argv) {
  /////////////////////////////////////////////////////////////
  /////////// Parsing the command line arguments //////////////
  /////////////////////////////////////////////////////////////
  Cmdline *cmd = parseCmdline(argc, argv);
  showOptionValues();

  int nfitsfiles = cmd->argc;
  char **fitsfiles = cmd->argv;

  printf("No. of infiles = %d\n", nfitsfiles);
  for (int i = 0; i < nfitsfiles; ++i) {
    printf("input file %d: %s \n", i, fitsfiles[i]);
  }

  if (cmd->numdms > 0) {
    if (cmd->numdms < 256) {
      std::cerr << "ERROR: Please specify at least 256 numdms" << std::endl;
      exit(1);
    }
    if (cmd->dmstep == 0) {
      std::cerr
          << "ERROR: Non zero dmstep value required when specifying numdms!"
          << std::endl;
      exit(1);
    }
  }
  ////////////////////////////////////////////////////////////
  //////////////// WARNING ///////////////////////////////////
  // Using only the first fits file for extracting the
  // relevant parameters. It is assumed implicitly that
  // the parameters are consistent across all fits files.
  const char *first_filename = fitsfiles[0];

  ////////////////////////////////////////////////////////////
  //// Extracting relevant parameters from the fits file /////
  ////////////////////////////////////////////////////////////

  fitsfile *ffptr;
  int status = 0;
  fits_open_file(&ffptr, first_filename, READONLY, &status);
  if (status != 0) {
    printf("Error in opening first fits file!\n");
    exit(1);
  }

  // Get number of HDUs
  int num_hdus;
  fits_get_num_hdus(ffptr, &num_hdus, &status);
  printf("No. of HDUs = %d\n", num_hdus);

  char comment[80];
  char telescope_name[100];
  fits_read_key(ffptr, TSTRING, "TELESCOP", telescope_name, comment, &status);
  printf("telescope name = %s \n", telescope_name);

  char instrument[100];
  fits_read_key(ffptr, TSTRING, "BACKEND", instrument, comment, &status);
  printf("instrument name = %s \n", instrument);

  char object_name[100];
  fits_read_key(ffptr, TSTRING, "SRC_NAME", object_name, comment, &status);
  printf("object name = %s \n", object_name);

  char right_ascension[100];
  fits_read_key(ffptr, TSTRING, "RA", right_ascension, comment, &status);
  printf("right_ascension = %s \n", right_ascension);

  char declination[100];
  fits_read_key(ffptr, TSTRING, "DEC", declination, comment, &status);
  printf("declination = %s \n", declination+1);

  char observed_by[100];
  fits_read_key(ffptr, TSTRING, "OBSERVER", observed_by, comment, &status);
  printf("observer = %s \n", observed_by);

  int stt_imjd, stt_smjd;
  double stt_offs;
  double epoch;

  /* Read and print the integer keyword STT_IMJD */
    if (fits_read_key(ffptr, TINT, "STT_IMJD", &stt_imjd, comment, &status)) {
        fits_report_error(stderr, status);
        return(status);
    }
    printf("STT_IMJD = %d   (%s)\n", stt_imjd, comment);

    /* Read and print the integer keyword STT_SMJD */
    if (fits_read_key(ffptr, TINT, "STT_SMJD", &stt_smjd, comment, &status)) {
        fits_report_error(stderr, status);
        return(status);
    }
    printf("STT_SMJD = %d   (%s)\n", stt_smjd, comment);

    /* Read and print the double keyword STT_OFFS */
    if (fits_read_key(ffptr, TDOUBLE, "STT_OFFS", &stt_offs, comment, &status)) {
        fits_report_error(stderr, status);
        return(status);
    }
    printf("STT_OFFS = %.15f   (%s)\n", stt_offs, comment);

    /* Compute the final epoch: epoch = STT_IMJD + (STT_SMJD + STT_OFFS)/86400 */
    epoch = stt_imjd + ((stt_smjd + stt_offs) / 86400.0);
    printf("Computed epoch = %.15f\n", epoch);

  char projid[100];
  /* Read and print the integer keyword STT_SMJD */
    if (fits_read_key(ffptr, TSTRING, "PROJID", projid, comment, &status)) {
        fits_report_error(stderr, status);
        return(status);
    }
    printf("projid = %s   (%s)\n", projid, comment);

  char dateobs[100];
  /* Read and print the integer keyword STT_SMJD */
    if (fits_read_key(ffptr, TSTRING, "DATE-OBS", dateobs, comment, &status)) {
        fits_report_error(stderr, status);
        return(status);
    }
    printf("dateobs = %s   (%s)\n", dateobs, comment);

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
  printf("offs, scal width = %ld\n", scal_offs_width);

  int naxis2;
  fits_read_key(ffptr, TINT, "NAXIS2", &naxis2, comment, &status);
  printf("naxis2 = %d\n", naxis2);

  int colnum = -1;
  fits_get_colnum(ffptr, 0, "DAT_FREQ", &colnum, &status);

  double *freqs = (double *)malloc(sizeof(double) * nchans_read);
  if (status == COL_NOT_FOUND) {
    printf("Warning!:  Can't find the channel freq column!\n");
    status = 0; // Reset status
  } else {
    int anynull = 0;
    fits_read_col(ffptr, TDOUBLE, colnum, 1L, 1L, nchans_read, 0, freqs,
                  &anynull, &status);
  }

  printf("lo, hi freq = %.15f, %.15f\n", freqs[0], freqs[nchans_read - 1]);

  fits_close_file(ffptr, &status);
  
  ///////////////////////////////////////////////////////////////////
  //////////////// Initializing more parameters /////////////////////
  ///////////////////////////////////////////////////////////////////

  dedisp_float dm_start = cmd->lodm;      // pc cm^-3
  dedisp_float dm_end = cmd->hidm;        // pc cm^-3
  dedisp_float pulse_width = cmd->pwidth; // ms
  dedisp_float downsamp = cmd->downsamp;

  int device_idx = 0;

  dedisp_float sampletime_base = tbin;
  printf("sampletime base = %f\n", sampletime_base);

  // Observation time for the whole timeseries distributed in multiple
  // files
  dedisp_float Tobs =
      tbin * nsblk * naxis2 * nfitsfiles; // Observation duration in seconds
  printf("Tobs = %f\n", Tobs);

  dedisp_float dt = downsamp * sampletime_base; // s
  printf("dt = %f\n", dt);

  dedisp_float f0 = freqs[nchans_read - 1]; // MHz (highest channel!)
  printf("f0 = %f\n", f0);

  dedisp_size nchans = nchans_read;

  // (hifreq-lofreq)/(nchans-1)
  double ddf = ((double)freqs[0] - (double)freqs[nchans_read - 1]) /
                    (nchans - 1);
  printf("double ddf = %.15f\n", ddf);

  dedisp_float df = ddf; // MHz   (This must be negative!)
  printf("df = %.15f\n", df);

  dedisp_float bw = ((double)freqs[nchans_read - 1] - (double)freqs[0]) /
                    (nchans - 1) * nchans; // MHz
  printf("bw = %f\n", bw);

  // nsamps scales with nfitsfiles because Tobs is scaled already
  dedisp_size nsamps = Tobs / dt;
  dedisp_float dm_tol = 1.25;
  dedisp_size in_nbits = 8;
  dedisp_size out_nbits =
      32; // DON'T CHANGE THIS FROM 32, since that signals it to use floats

  dedisp_size dm_count;
  dedisp_size max_delay;
  dedisp_byte *input = 0;
  dedisp_float *output = 0;

  const dedisp_float *dmlist;

  clock_t startclock;

  /////////////////////////////////////////////////////////////
  //////////////// Finding byte size of HDUs //////////////////
  /////////////////////////////////////////////////////////////
  //////////// Assuming consistency in all fits files /////////

  FILE *fptr;

  if ((fptr = fopen(first_filename, "rb")) == NULL) {
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
  //long chunksize = 2147479552; 
  long chunksize = 1073741824;

  // Creating buffer to hold all file contents
  unsigned char *table_full;
  if (posix_memalign((void **)&table_full, 4096, file_size_aligned) != 0) {
    perror("posix_memalign");
    return -1;
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();

  int poln = 0;

  float *rawdata;
  // Allocating enough memory to hold the reduced data from all files
  rawdata = (float *)calloc((size_t)nsblk * naxis2 * nchans * nfitsfiles,
                            sizeof(float));

  // Looping over the files
  for (int file_idx = 0; file_idx < nfitsfiles; ++file_idx) {
    const char *filename = fitsfiles[file_idx];
    // Opening file DIRECT and no DIRECT
    int fd = open(filename, O_RDONLY | O_DIRECT);
    int fd_nodirect = open(filename, O_RDONLY);

    if (fd == -1) {
      perror("open");
      return -1;
    }

    // Reading the whole file and timing the read
    start_time = std::chrono::high_resolution_clock::now();
    long chunks =
        getDataFromRows(fd, table_full, chunksize, file_size, fd_nodirect);
    end_time = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();

    long megabytes_read = (double)(file_size) / 1e6;
    std::cout << "read " << chunks << " chunks with chunksize = " << chunksize
              << " from file " << fitsfiles[file_idx]
              << ", time: " << (double)duration_us / 1e6 << " seconds"
              << ", Read speed (MB/s): "
              << megabytes_read / (double)duration_us * 1e6 << std::endl;

    float *current_rawdata_ptr =
        rawdata + (size_t)nsblk * naxis2 * nchans * file_idx;

    start_time = std::chrono::high_resolution_clock::now();
    reduceBinaryTable(table_full, current_rawdata_ptr, poln, naxis1, naxis2,
                      nsblk, nchans, npol, offset_for_data, initial_offset,
                      scal_offs_width);

    end_time = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();

    std::cout << "reduction finished, time: " << (double)duration_us / 1e6
              << std::endl;

    close(fd);
    close(fd_nodirect);
  }

  free(table_full);

  /*
     input is a pointer to an array containing a time series of length
     nsamps for each frequency channel in plan. The data must be in
     time-major order, i.e., frequency is the fastest-changing
     dimension, time the slowest. There must be no padding between
     consecutive frequency channels.
   */

  // input = (dedisp_byte *)malloc(nsamps * nchans * (in_nbits / 8));
  input = (dedisp_byte *)rawdata;

  printf("Create plan and init GPU\n");
  // Create a dedispersion plan
  PlanType plan(nchans, dt, f0, df, device_idx);

  printf("Gen DM list\n");
  // Generate a list of dispersion measures for the plan
  if (cmd->numdms == 0) {
    std::cout << "Numdms not specified, generating DM list using the internal "
                 "function"
              << std::endl;
    plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tol);
  } else {
    std::cout << "Generating equispaced DM list using the provided step size"
              << std::endl;
    plan.generate_dm_list_equispaced(cmd->lodm, cmd->dmstep, cmd->numdms);
  }

  // Find the parameters that determine the output size
  dm_count = plan.get_dm_count();
  max_delay = plan.get_max_delay();
  dmlist = plan.get_dm_list();
  // dt_factors = plan.get_dt_factors(plan);

  printf("----------------------------- DM COMPUTATIONS  "
         "----------------------------\n");
  printf("Computing %lu DMs from %f to %f pc/cm^3\n", dm_count, dmlist[0],
         dmlist[dm_count - 1]);
  printf("Max DM delay is %lu samples (%.3f seconds)\n", max_delay,
         max_delay * dt);
  std::cout << "dt = " << dt << std::endl;

  // nsamp_fft is the fft size. Keeping this bigger than nsamps is implicitly
  // adding zero padding to the end of the timeseries. Choosing nsamps + max_delay
  // as nsamps_fft prevents contamination at the ends when combined with shifts to 
  // right. The time samples even when missing info from channels remain chronologically
  // relevant.

  // nsamp_padded is chosen large enough to hold complex coefficients arising 
  // from fourier transforms. 

  //nsamp_fft = dedisp::round_up(nsamps + max_delay, 16384);
  nsamp_fft = closestOptimal(nsamps + max_delay,true);
  //nsamp_fft = nsamps;
  nsamp_padded = 2ULL * (nsamp_fft/2 + 1);

  // output clean timeseries
  if (cmd->cleanoutP) {
      nsamps_computed = nsamps - max_delay;
  }
  // output dirty timeseries
  else {
      //nsamps_computed = nsamps;
      //nsamps_computed = nsamp_fft;
      nsamps_computed = nsamps + max_delay;
  }

  // Make nsamps_computed even
  nsamps_computed = (nsamps_computed/2) * 2; 

  if (cmd->fftoutP) {
    printf("Computing %lu Fourier Coefficients of dedispersed timeseries "
          "(adjusting for max delay)\n",
          nsamp_fft);
    printf("Output data array size : %lu MB\n",
          (dm_count * nsamp_fft * (out_nbits / 8)) / (1 << 20));

    // Output is chosen such that it is able to hold all the FFT coefficients
    output = (dedisp_float *)malloc(nsamp_padded * dm_count * out_nbits / 8);
  }
  else {
    printf("Computing %lu out of %lu total samples (%.2f%% efficiency)\n",
         nsamps_computed, nsamps,
         100.0 * (dedisp_float)nsamps_computed / nsamps);
    printf("Output data array size : %lu MB\n",
          (dm_count * nsamps_computed * (out_nbits / 8)) / (1 << 20));
    output = (dedisp_float *)malloc(nsamps_computed * dm_count * out_nbits / 8);
  }

  printf("\n");

  if (output == NULL) {
    printf("\nERROR: Failed to allocate output array\n");
    return -1;
  }

  printf("Compute on GPU\n");
  aa_gpu_timer timer;
  timer.Start();
  // Compute the dedispersion transform on the GPU
  plan.execute(nsamps, input, in_nbits, (dedisp_byte *)output, out_nbits);
  timer.Stop();
  printf("plan.execute() took %.2f seconds\n", timer.Elapsed());

#if WRITE_INPUT_DATA
  FILE *file_in = fopen("input.bin", "wb");
  fwrite(input, 1, (size_t)nsamps * nchans * (in_nbits / 8), file_in);
  fclose(file_in);
#endif

  const char* outfiles_basename = (cmd->outfile == NULL) ? "output" : cmd->outfile;

  if (cmd->multoutP) {
    start_time = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for 
    for (unsigned int out_file_idx = 0 ; out_file_idx < dm_count ; ++out_file_idx) {
      char out_file_name[256], out_inf_name[256];
      sprintf(out_file_name,"%s_%ld_DM%.2f.%s", outfiles_basename, out_file_idx, dmlist[out_file_idx], (cmd->fftoutP)? "fft":"dat");
      sprintf(out_inf_name,"%s_%ld_DM%.2f.inf", outfiles_basename, out_file_idx, dmlist[out_file_idx]);
      
      FILE* file_out = fopen(out_file_name, "wb");
      size_t numtowrite = (size_t)(cmd->fftoutP? nsamp_padded : nsamps_computed) * out_nbits / 8;

      size_t writtennum = fwrite(output + out_file_idx * (size_t)(cmd->fftoutP? nsamp_padded : nsamps_computed), 
            1, 
            numtowrite, 
            file_out);

      if (writtennum != numtowrite) {
        std::cerr << "Writing file " << out_file_idx << " failed!" << std::endl;
      }

      fclose(file_out);

      FILE* inf_out = fopen(out_inf_name,"w");

      // Writing the inf data
      fprintf(inf_out,"%-40s=  %s_DM_%.2f\n", " Data file name without suffix" ,outfiles_basename, dmlist[out_file_idx]);
      fprintf(inf_out,"%-40s=  %s\n", " Telescope used", telescope_name);
      fprintf(inf_out,"%-40s=  %s\n", " Instrument used", instrument);
      fprintf(inf_out,"%-40s=  %s\n", " Object being observed", object_name);
      fprintf(inf_out,"%-40s=  %s\n", " J2000 Right Ascension (hh:mm:ss.ssss)", right_ascension);
      fprintf(inf_out,"%-40s=  %s\n", " J2000 Declination     (dd:mm:ss.ssss)", declination + 1);
      fprintf(inf_out,"%-40s=  %s\n", " Data observed by", observed_by);
      fprintf(inf_out,"%-40s=  %.15f\n", " Epoch of observation (MJD)", epoch);
      fprintf(inf_out,"%-40s=  0\n", " Barycentered?           (1 yes, 0 no)", observed_by);
      fprintf(inf_out,"%-40s=  %ld\n", " Number of bins in the time series", nsamps);
      fprintf(inf_out,"%-40s=  %.4f\n", " Width of each time series bin (sec)", dt);
      fprintf(inf_out,"%-40s=  1\n", " Any breaks in the data? (1 yes, 0 no)");
      fprintf(inf_out,"%-40s=  0, %ld\n", " On/Off bin pair #  1 ", nsamps_computed-1);
      fprintf(inf_out,"%-40s=  %ld, %ld\n", " On/Off bin pair #  2", nsamps-1, nsamps-1);
      fprintf(inf_out,"%-40s=  Radio\n", " Type of observation (EM band)  ");
      fprintf(inf_out,"%-40s=  900\n", " Beam diameter (arcsec)");
      fprintf(inf_out,"%-40s=  %.2f\n", " Dispersion measure (cm-3 pc)", dmlist[out_file_idx]);
      fprintf(inf_out,"%-40s=  %.7f\n", " Central freq of low channel (MHz)", freqs[0]);
      fprintf(inf_out, "%-40s=  %.7f\n", " Total bandwidth (MHz)", bw);
      fprintf(inf_out, "%-40s=  %d\n", " Number of channels", nchans);
      fprintf(inf_out, "%-40s=  %.15f\n", " Channel bandwidth (MHz)", -ddf);

      char *user = getenv("USER");  // Get the username
      if (!user) user = getenv("USERNAME");  // Fallback for Windows

      fprintf(inf_out, "%-40s=  %s\n", " Data analyzed by", user ? user : "Unknown");
      fprintf(inf_out, " Any additional notes: \n \tProject ID %s, Date: 2%s.\n \t4 polns were not summed.  Samples have 8 bits. \n", projid, dateobs);
      
      fclose(inf_out);
      
    }

    end_time = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();
    std::cout << "Writing the output data in multiple files took " << (double)duration_us / 1e6
              << " seconds" << std::endl;
  } else {
    printf("abc");
    FILE *file_out;
    if (cmd->fftoutP) {
      char out_file_name[256];
      sprintf(out_file_name,"%s.allDMs.fft", outfiles_basename);
      file_out = fopen(out_file_name, "wb");
    } else {
      char out_file_name[256];
      sprintf(out_file_name,"%s.allDMs.dat", outfiles_basename);
      file_out = fopen(out_file_name, "wb");
    }

    start_time = std::chrono::high_resolution_clock::now();
    fwrite(output, 1, (size_t)(cmd->fftoutP? nsamp_padded : nsamps_computed) * dm_count * out_nbits / 8,
          file_out);
    fclose(file_out);
    end_time = std::chrono::high_resolution_clock::now();
    duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                      end_time - start_time)
                      .count();
    std::cout << "Writing the output data took " << (double)duration_us / 1e6
              << " seconds" << std::endl;
  }

  // Clean up
  free(output);
// free(input);
#ifdef READFROMFILE
  free(rawdata);
#endif
  printf("Dedispersion successful.\n");
  return 0;
}