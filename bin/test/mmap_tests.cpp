#include "hip/hip_runtime.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <fcntl.h>
#include <unistd.h>

#include "fdd_gpu.h"

#include <Plan.hpp>
#include <hip/hip_runtime.h>

#include "fdd/helper.h"

dedisp_float maxval_data = std::numeric_limits<float>::lowest();
dedisp_float minval_data = std::numeric_limits<float>::max();

//static unsigned char read_buffer[123002956]

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

void reduceData_mapped(float *reduceddata, unsigned char *mapped, int poln, off_t missing_offset) {
    for (int subint = 0 ; subint < 200 ; ++subint) {
        unsigned char* data_wts = mapped + subint * 123002956 + 12364 + missing_offset;
        unsigned char* data_offs = data_wts + 12288;
        unsigned char* data_scl = data_offs + 49152;
        unsigned char* rawdata = data_scl + 49152;
        for (int spectra = 0; spectra < 10000; ++spectra) {
            for (int chan = 0; chan < 3072; ++chan) {
            // Hard coded for zero_off = 0
            reduceddata[10000L * 3072L * subint + 3072L * spectra + chan] =
                ((float)rawdata[3072L * 4 * spectra + 3072L * poln + chan] *
                    data_scl[3072L * poln + chan] +
                data_offs[3072L * poln + chan]) *
                data_wts[chan];

            /* minval_data = std::min(
                reduceddata[10000L * 3072L * subint + 3072L * spectra + chan],
                minval_data);
            maxval_data = std::max(
                reduceddata[10000L * 3072L * subint + 3072L * spectra + chan],
                maxval_data); */
            }
        }
    }
  
}

// run method for dedispersion with original dedisp test implementation
int main(int argc, char **argv) {
  int device_idx = 0;
  
  dedisp_float sampletime_base =
      100.0E-6; 

  dedisp_float downsamp = 1.0;
  dedisp_float Tobs = 200.0; // Observation duration in seconds
  dedisp_float dt = downsamp * sampletime_base; // s (0.25 ms sampling)
  dedisp_float f0 = 169.589996337891;           // MHz (highest channel!)
  dedisp_size nchans = 3072;
  dedisp_float bw = 30.7199914522957/3072. * nchans;           // MHz
  dedisp_float df = -1.0 * bw / nchans; // MHz   (This must be negative!)

  dedisp_size nsamps = Tobs / dt;

  /* dedisp_float dm_start = 0.0;    // pc cm^-3
  dedisp_float dm_end = 10.0;     // pc cm^-3
  dedisp_float pulse_width = 4.0; // ms */

  dedisp_float dm_tol = 1.25;
  dedisp_size in_nbits = 8;
  dedisp_size out_nbits =
      32; // DON'T CHANGE THIS FROM 32, since that signals it to use floats
  dedisp_byte *input = 0;
  dedisp_float *output = 0;

  unsigned int i, nc, ns, nd;


  clock_t startclock;

  /*
    Reading the data from fits file without using the cfitsio lib
    The way it is read is very hard coded
  */

  const char *filename =
      "/home/pp/mwa_data/G0057_1368033096_15:43:38.82_+09:29:16.30_ch109-132_0001.fits"; 

  int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        perror("open");
        return -1;
    }

  // Map the binary table data
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting file size");
        close(fd);
        return EXIT_FAILURE;
    }

  off_t binary_table_offset = 5 * 2880;
  long page_size = sysconf(_SC_PAGESIZE);
  off_t aligned_offset = (binary_table_offset / page_size) * page_size;

  off_t missing_offset = binary_table_offset - aligned_offset;

  size_t map_size = sb.st_size - aligned_offset;

  printf("map_size: %ld, fd: %d, offset: %ld, aligned offset: %ld\n", map_size, fd, binary_table_offset, aligned_offset);
  void *mapped_void = mmap(NULL, map_size, PROT_READ, MAP_PRIVATE, fd, aligned_offset);

  printf("page_size = %ld \n",page_size);
  if (mapped_void == MAP_FAILED) {
        perror("Error mapping file");
        close(fd);
        return EXIT_FAILURE;
    }

  unsigned char* mapped = (unsigned char*) mapped_void;

  char smallbuffer[80];

  double zero_off = 0;

  /* for (int i = 0; i < 5; ++i) {
    printf("size 2880 block no. %d\n", i);
    for (int j = 0; j < 36; ++j) {
      //fread(smallbuffer, 1, 80, fptr);
      read(fd, smallbuffer, 80);
      printf("%s\n", smallbuffer);
    }
  } */

  // size of data in 1 row (8 bit data)
  unsigned char *rawdata_full =
      (unsigned char *)calloc(122880000, sizeof(unsigned char));
  float *data_scl = (float *)calloc(12288, sizeof(float));
  float *data_offs = (float *)calloc(12288, sizeof(float));
  float *data_wts = (float *)calloc(3072, sizeof(float));

  unsigned char *buffer =
      (unsigned char *)calloc(123002956, sizeof(unsigned char));

  int poln_to_use = 0;

  dedisp_float *rawdata;
  rawdata = (float *)calloc(10000L * 200L * 3072L, sizeof(float));

  auto start_time = std::chrono::high_resolution_clock::now();

  reduceData_mapped(rawdata, mapped, 0, missing_offset);
  /* for (int subint = 0; subint < 200; ++subint) {
    getDataFromRow_buffered(fd, rawdata_full, data_scl, data_offs, data_wts, subint, buffer);
    //getDataFromRow(fd, rawdata_full, data_scl, data_offs, data_wts, subint);
    reduceData(rawdata, rawdata_full, data_scl, data_offs, data_wts, 0, subint);
  } */
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();
  std::cout << "Reading all the subints took " << (double)duration_us / 1e6
            << " seconds" << std::endl;
  std::cout << "Minimum value: " << minval_data
            << ", maximum value: " << maxval_data << std::endl; 

  close(fd);
  return 0;
  
}