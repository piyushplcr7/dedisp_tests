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

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <random>

#include <Plan.hpp>
#include <cuda_runtime.h>

// Debug options
#define WRITE_INPUT_DATA 0
#define WRITE_OUTPUT_DATA 0

dedisp_float maxval_data = std::numeric_limits<float>::lowest();
dedisp_float minval_data = std::numeric_limits<float>::max();

struct aa_gpu_timer {
  cudaEvent_t start;
  cudaEvent_t stop;

  aa_gpu_timer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~aa_gpu_timer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start, 0); }

  void Stop() { cudaEventRecord(stop, 0); }

  float Elapsed() {
    float elapsed = 0.0;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed / 1000.0f;
  }
};

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
                    float *data_offs, float *data_wts, int subint) {
  // Move to dat_wts col
  fseek(fptr, 12364, SEEK_CUR);

  // read dat_wts
  fread(data_wts, 4, 3072, fptr);
  // read data_offs
  fread(data_offs, 4, 12288, fptr);
  // read data_scl
  fread(data_scl, 4, 12288, fptr);
  // read data
  fread(rawdata, 1, 122880000, fptr);

  for (int i = 0; i < 12288; ++i) {
    swapEndian(&data_offs[i]);
    swapEndian(&data_scl[i]);
  }

  for (int i = 0; i < 3072; ++i) {
    swapEndian(&data_wts[i]);
  }
}

void reduceData(float *reduceddata, unsigned char *rawdata, float *data_scl,
                float *data_offs, float *data_wts, int poln, int subint) {
  for (int spectra = 0; spectra < 10000; ++spectra) {
    for (int chan = 0; chan < 3072; ++chan) {
      // Hard coded for zero_off = 0
      reduceddata[10000L * 3072L * subint + 3072L * spectra + chan] =
          ((float)rawdata[3072L * 4 * spectra + 3072L * poln + chan] *
               data_scl[3072L * poln + chan] +
           data_offs[3072L * poln + chan]) *
          data_wts[chan];

      minval_data = std::min(
          reduceddata[10000L * 3072L * subint + 3072L * spectra + chan],
          minval_data);
      maxval_data = std::max(
          reduceddata[10000L * 3072L * subint + 3072L * spectra + chan],
          maxval_data);
    }
  }
}

// Assume input is a 0 mean float and quantize to an unsigned 8-bit quantity
dedisp_byte bytequant(dedisp_float f, dedisp_float minval,
                      dedisp_float maxval) {
  // initial range: [minval , maxval] <-> [a, b]
  // final range [-127.5 , 127.5] <-> [c, d]
  dedisp_float m = 255.0f / (maxval - minval);
  dedisp_float n = -127.5f * (maxval + minval) / (maxval - minval);
  // Bring the float value to the right range
  dedisp_float v = m * f + n + 127.5f;
  dedisp_byte r = (dedisp_byte)roundf(v);
  return r;
}

// Assume input is a 0 mean float and quantize to an unsigned 8-bit quantity
dedisp_byte bytequant_old(dedisp_float f) {
  dedisp_float v = f + 127.5f;
  dedisp_byte r;
  if (v > 255.0) {
    r = (dedisp_byte)255;
  } else if (v < 0.0f) {
    r = (dedisp_byte)0;
  } else {
    r = (dedisp_byte)roundf(v);
  }
  // printf("ROUND %f, %u\n",f,r);
  return r;
}

// Compute mean and standard deviation of an unsigned 8-bit array
void calc_stats_8bit(dedisp_byte *a, dedisp_size n, dedisp_float *mean,
                     dedisp_float *sigma) {
  // Use doubles to prevent rounding error
  double sum = 0.0, sum2 = 0.0;
  double mtmp = 0.0, vartmp;
  double v;
  dedisp_size i;

  // Use corrected 2-pass algorithm from Numerical Recipes
  sum = 0.0;
  for (i = 0; i < n; i++) {
    v = (double)a[i];
    sum += v;
  }
  mtmp = sum / n;

  sum = 0.0;
  sum2 = 0.0;
  for (i = 0; i < n; i++) {
    v = (double)a[i];
    sum2 += (v - mtmp) * (v - mtmp);
    sum += v - mtmp;
  }
  vartmp = (sum2 - (sum * sum) / n) / (n - 1);
  *mean = mtmp;
  *sigma = sqrt(vartmp);

  return;
}

// Compute mean and standard deviation of a float array
void calc_stats_float(dedisp_float *a, dedisp_size n, dedisp_float *mean,
                      dedisp_float *sigma) {
  // Use doubles to prevent rounding error
  double sum = 0.0, sum2 = 0.0;
  double mtmp = 0.0, vartmp;
  double v;
  dedisp_size i;

  // Use corrected 2-pass algorithm from Numerical Recipes
  sum = 0.0;
  for (i = 0; i < n; i++) {
    sum += a[i];
  }
  mtmp = sum / n;

  sum = 0.0;
  sum2 = 0.0;
  for (i = 0; i < n; i++) {
    v = a[i];
    sum2 += (v - mtmp) * (v - mtmp);
    sum += v - mtmp;
  }
  vartmp = (sum2 - (sum * sum) / n) / (n - 1);
  *mean = mtmp;
  *sigma = sqrt(vartmp);

  return;
}

// run method for dedispersion with original dedisp test implementation
template <typename PlanType> int run() {
  int device_idx = 0;

  dedisp_float sampletime_base =
      0.1; //100.0E-6; // Base is 250 microsecond time samples
  dedisp_float downsamp = 1.0;
  dedisp_float Tobs = 2.60001; //200; Observation duration in seconds
  dedisp_float dt = downsamp * sampletime_base; // s (0.25 ms sampling)
  dedisp_float f0 = 1e6;//169.589996337891;           // MHz (highest channel!)
  dedisp_float bw =  1e5;//30.7199914522957;           // MHz
  dedisp_size nchans = 4; // 3072
  dedisp_float df = -1.0 * bw / nchans; // MHz   (This must be negative!)

  dedisp_size nsamps = Tobs / dt;
  std::cout << "Tobs = " << Tobs << ", dt = " << dt << std::endl;
  std::cout << "nsamps = " << nsamps << std::endl;

  dedisp_float dm_start = 0.0;    // pc cm^-3
  dedisp_float dm_end = 200.0;     // pc cm^-3
  dedisp_float pulse_width = 4.0; // ms
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

  /*
    Reading the data from fits file without using the cfitsio lib
    The way it is read is very hard coded
  */
#ifdef READFROMFILE
  const char *filename =
      "/home/pp/G0057_1368033096_15:43:38.82_+09:29:16.30_ch109-132_0001.fits";

  FILE *fptr;

  if ((fptr = fopen(filename, "rb")) == NULL) {
    printf("Error! opening file");

    // Program exits if the file pointer returns NULL.
    exit(1);
  }

  char smallbuffer[80];

  double zero_off = 0;

  for (int i = 0; i < 5; ++i) {
    printf("size 2880 block no. %d\n", i);
    for (int j = 0; j < 36; ++j) {
      fread(smallbuffer, 1, 80, fptr);
      printf("%s\n", smallbuffer);
    }
  }

  // size of data in 1 row (8 bit data)
  unsigned char *rawdata_full =
      (unsigned char *)calloc(122880000, sizeof(unsigned char));
  float *data_scl = (float *)calloc(12288, sizeof(float));
  float *data_offs = (float *)calloc(12288, sizeof(float));
  float *data_wts = (float *)calloc(3072, sizeof(float));

  int poln_to_use = 0;

  dedisp_float *rawdata;
  rawdata = (float *)calloc(10000L * 200L * 3072L, sizeof(float));

  auto start_time = std::chrono::high_resolution_clock::now();
  for (int subint = 0; subint < 200; ++subint) {
    getDataFromRow(fptr, rawdata_full, data_scl, data_offs, data_wts, subint);
    reduceData(rawdata, rawdata_full, data_scl, data_offs, data_wts, 0, subint);
  }
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();
  std::cout << "Reading all the subints took " << (double)duration_us / 1e6
            << " seconds" << std::endl;
  std::cout << "Minimum value: " << minval_data
            << ", maximum value: " << maxval_data << std::endl;
  /* std::cout << "Reading all the subints took "
            << std::chrono::duration_cast<std::chrono::seconds>(end_time -
                                                                start_time)
                   .count()
            << std::endl; */

  fclose(fptr);
#endif

  // Synthetic data for testing
  float *rawdata = (float*) calloc(nsamps * nchans,sizeof(float));
  // Filling the synthetic data
  for (int i = 0 ; i < nsamps ; ++i) {
    for (int j = 0 ; j < nchans ; ++j) {
      // channels is fastest changing
      rawdata[i*nchans+j] = std::sin(M_PI * i * dt);
      /* if (j == 0)
        rawdata[i*nchans+j] = std::sin(M_PI * i * dt);
      else
        rawdata[i*nchans+j] = std::cos(M_PI * i * dt); */
    }
  }

  std::cout << "Time series all channels:" << std::endl;
  for (int i = 0 ; i < nchans ; ++i) {
    for (int j = 0 ; j < nsamps ; ++j) {
      std::cout << rawdata[j * nchans + i] << ", " ;
    }
    std::cout << std::endl << std::endl;
  }

  /* std::cout << "time series channel 0:" << std::endl;
  for (int i = 0 ; i < nsamps ; ++i) {
    for (int j = 0 ; j < nchans ; ++j) {
      // channels is fastest changing
      if (j == 0)
        std::cout << rawdata[i*nchans+j]  << ", " ;
    }
  }
  std::cout << std::endl;

  std::cout << "time series channel 1:" << std::endl;
  for (int i = 0 ; i < nsamps ; ++i) {
    for (int j = 0 ; j < nchans ; ++j) {
      // channels is fastest changing
      if (j == 1)
        std::cout << rawdata[i*nchans+j]  << ", " ;
    }
  }
  std::cout << std::endl; */

  maxval_data = 1;
  minval_data = -1;

  /*
     input is a pointer to an array containing a time series of length
     nsamps for each frequency channel in plan. The data must be in
     time-major order, i.e., frequency is the fastest-changing
     dimension, time the slowest. There must be no padding between
     consecutive frequency channels.
   */
  dedisp_float raw_mean, raw_sigma;
  calc_stats_float(rawdata, nsamps * nchans, &raw_mean, &raw_sigma);
  printf("Rawdata Mean (includes signal)    : %f\n", raw_mean);
  printf("Rawdata StdDev (includes signal)  : %f\n", raw_sigma);

  input = (dedisp_byte *)malloc(nsamps * nchans * (in_nbits / 8));

  printf("Quantizing array\n");
  /* Now fill array by quantizing rawdata */
  for (ns = 0; ns < nsamps; ns++) {
    for (nc = 0; nc < nchans; nc++) {
      input[ns * nchans + nc] =
          bytequant(rawdata[ns * nchans + nc], minval_data, maxval_data);
    }
  }

  std::cout << "Quantized time series all channels:" << std::endl;
  for (int i = 0 ; i < nchans ; ++i) {
    for (int j = 0 ; j < nsamps ; ++j) {
      std::cout << (int)input[j * nchans + i] << ", " ;
    }
    std::cout << std::endl << std::endl;
  }

  /**/
  // Writing the data to binary files
  FILE *fptr_out_float;

  if ((fptr_out_float = fopen("rawdata.bin", "wb")) == NULL) {
    printf("Error! opening file for float output");

    // Program exits if the file pointer returns NULL.
    exit(1);
  }

  fwrite(rawdata, sizeof(float), (size_t) nsamps * nchans, fptr_out_float);
  fclose(fptr_out_float);

  FILE *fptr_out_byte;

  if ((fptr_out_byte = fopen("input.bin", "wb")) == NULL) {
    printf("Error! opening file for float output");

    // Program exits if the file pointer returns NULL.
    exit(1);
  }

  fwrite(input, sizeof(unsigned char), (size_t) nsamps * nchans, fptr_out_byte);
  fclose(fptr_out_byte);
  // exit(1);
  /**/

  /* for (int i = 20000; i < 20020; ++i) {
    printf("i=%5d %9.3f %9.3f \n", i, rawdata[i], (float)input[i]);
  } */

  dedisp_float in_mean, in_sigma;
  calc_stats_8bit(input, nsamps * nchans, &in_mean, &in_sigma);

  printf("Quantized data Mean (includes signal)    : %f\n", in_mean);
  printf("Quantized data StdDev (includes signal)  : %f\n", in_sigma);
  printf("\n");

  printf("Create plan and init GPU\n");
  // Create a dedispersion plan
  PlanType plan(nchans, dt, f0, df, device_idx);

  printf("Gen DM list\n");
  // Generate a list of dispersion measures for the plan
  plan.generate_dm_list(dm_start, dm_end, pulse_width, dm_tol);

  // Find the parameters that determine the output size
  dm_count = plan.get_dm_count();
  max_delay = plan.get_max_delay();
  nsamps_computed = nsamps - max_delay;
  dmlist = plan.get_dm_list();
  // dt_factors = plan.get_dt_factors(plan);

  printf("----------------------------- DM COMPUTATIONS  "
         "----------------------------\n");
  printf("Computing %lu DMs from %f to %f pc/cm^3\n", dm_count, dmlist[0],
         dmlist[dm_count - 1]);
  printf("Max DM delay is %lu samples (%.f seconds)\n", max_delay,
         max_delay * dt);
  printf("Computing %lu out of %lu total samples (%.2f%% efficiency)\n",
         nsamps_computed, nsamps,
         100.0 * (dedisp_float)nsamps_computed / nsamps);
  printf("Output data array size : %lu MB\n",
         (dm_count * nsamps_computed * (out_nbits / 8)) / (1 << 20));
  printf("\n");

  // Allocate space for the output data
  output = (dedisp_float *)malloc(nsamps_computed * dm_count * out_nbits / 8);
  if (output == NULL) {
    printf("\nERROR: Failed to allocate output array\n");
    return -1;
  }

  printf("Compute on GPU\n");
  startclock = clock();
  aa_gpu_timer timer;
  timer.Start();
  // Compute the dedispersion transform on the GPU
  plan.execute(nsamps, input, in_nbits, (dedisp_byte *)output, out_nbits);
  timer.Stop();
  printf("plan.execute() took %.2f seconds\n", timer.Elapsed());
  printf("Dedispersion took %.2f seconds\n",
         (double)(clock() - startclock) / CLOCKS_PER_SEC);

  // Look for significant peaks
  /* dedisp_float out_mean, out_sigma;
  calc_stats_float(output, nsamps_computed * dm_count, &out_mean, &out_sigma);

  printf("Output RMS                               : %f\n", out_mean);
  printf("Output StdDev                            : %f\n", out_sigma);

  std::cout << "nsamps_computed: " << nsamps_computed
            << ", dm_count: " << dm_count << std::endl;

  i = 0;
  for (nd = 0; nd < dm_count; nd++) {
    for (ns = 0; ns < nsamps_computed; ns++) {
      dedisp_size idx = nd * nsamps_computed + ns;
      dedisp_float val = output[idx];
      if (val - out_mean > 6.0 * out_sigma) {
        printf(
            "DM trial %u (%.3f pc/cm^3), Samp %u (%.6f s): %f (%.2f sigma)\n",
            nd, dmlist[nd], ns, ns * dt, val, (val - out_mean) / out_sigma);
        i++;
        if (i > 100)
          break;
      }
    }
    if (i > 100)
      break;
  } */

#if WRITE_INPUT_DATA
  FILE *file_in = fopen("input.bin", "wb");
  fwrite(input, 1, (size_t)nsamps * nchans * (in_nbits / 8), file_in);
  fclose(file_in);
#endif

  // #if WRITE_OUTPUT_DATA
  FILE *file_out = fopen("output.bin", "wb");
  fwrite(output, 1, (size_t)nsamps_computed * dm_count * out_nbits / 8,
         file_out);
  fclose(file_out);
  // #endif

  // Clean up
  free(output);
  free(input);
  /* free(rawdata_full);
  free(data_scl);
  free(data_offs);
  free(data_wts); */
  free(rawdata);
  printf("Dedispersion successful.\n");
  return 0;
}