/*
  Simple test application for libdedisp
  By Paul Ray (2013)
  With extended run method to use multiple different implementations
  (Dedisp PlanType) of dedispersion. (2020 ASTRON)
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include "fdd_gpu.h"
#include "fitsio.h"

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

#define READFROMFILE
#define WRITEFILES

int run(int argc, char **argv) {
  Cmdline *cmd = parseCmdline(argc, argv);
  showOptionValues();

  const char *filename = *cmd->argv;

  if (cmd->numdms >0) {
    if (cmd->dmstep == 0) {
      std::cerr << "ERROR: Non zero dmstep value required when specifying numdms!" << std::endl;
      exit(1);
    }
  }

  // Extracting relevant parameters from the fits file
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

  unsigned int i, nc, ns, nd;
  const dedisp_float *dmlist;

  clock_t startclock;
  #ifdef READFROMFILE

  FILE *fptr;

  if ((fptr = fopen(filename, "rb")) == NULL) {
    printf("Error! opening file");
    exit(1);
  }

  char smallbuffer[80];

  double zero_off = 0;

  for (int i = 0; i < 1000; ++i) {
    if (num_hdus == 0) {
      break;
    }
    for (int j = 0; j < 36; ++j) {
      fread(smallbuffer, 1, 80, fptr);
      // Find END 
      char* tempptr = strstr(smallbuffer, "END     ");
      if (tempptr != NULL) {
        num_hdus--;
      }
    }
  }

  // size of data in 1 row (8 bit data)
  unsigned char *rawdata_full =
      (unsigned char *)calloc(data_byte_width, sizeof(unsigned char));
  float *data_scl = (float *)calloc(scal_offs_width, sizeof(float));
  float *data_offs = (float *)calloc(scal_offs_width, sizeof(float));
  float *data_wts = (float *)calloc(nchans, sizeof(float));

  int poln_to_use = 0;

  dedisp_float *rawdata;
  rawdata = (float *)calloc((size_t)nsblk * naxis2 * nchans, sizeof(float));

  dedisp_float *rawdata_transposed;
  rawdata_transposed = (float *)calloc((size_t)nsblk * naxis2 * nchans, sizeof(float));

  // Initial offset 
  size_t initial_offset = (size_t)naxis1 - 4*nchans - 2 * 4 * scal_offs_width - data_byte_width;
  printf("initial offset = %ld\n", initial_offset);

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int subint = 0; subint < naxis2; ++subint) {
    getDataFromRow(fptr, rawdata_full, data_scl, 
                   data_offs, data_wts, subint, 
                   data_byte_width, scal_offs_width, nchans, initial_offset);
    reduceData(rawdata, rawdata_full, data_scl, data_offs, data_wts, 0, subint,
               nsblk, nchans, npol); 
    reduceDataTransposed(rawdata_transposed, rawdata_full, data_scl, data_offs, data_wts, 0, subint,
               nsblk, nchans, npol, naxis2);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();
  std::cout << "Reading all the subints took " << (double)duration_us / 1e6
            << " seconds" << std::endl;
  std::cout << "Minimum value: " << minval_data
            << ", maximum value: " << maxval_data << std::endl; 

  fclose(fptr);

  #endif

  double tol = 1e-13;
  bool match = true;
  // Comparing the data
  for (size_t CHAN = 0 ; CHAN < nchans ; ++CHAN) {
    for (size_t SAMPLE = 0 ; SAMPLE < (size_t)nsblk * naxis2 ; ++SAMPLE) {
      if (fabs(rawdata[SAMPLE* nchans + CHAN] - rawdata_transposed[CHAN * nsblk * naxis2 + SAMPLE]) > tol ) {
        std::cout << "MISMATCH at chan = " << CHAN << ", SAMPLE = " << SAMPLE << std::endl;
        match = false;
      }
    }
  }
  
  std::cout << (match? "Match" : "Mismatch") << std::endl;
  
  #ifdef READFROMFILE
  free(rawdata_full);
  free(data_scl);
  free(data_offs);
  free(data_wts);
  free(rawdata);
  #endif
  return 0;
}