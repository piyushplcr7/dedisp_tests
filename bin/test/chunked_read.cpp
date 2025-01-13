
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
#include <fcntl.h>
#include <unistd.h>

#include "fdd_gpu.h"

#include <Plan.hpp>
#include <cuda_runtime.h>

#include "fdd/helper.h"

static int rows = 200;
static long bytes_per_row = 123002956;
static long bytes_to_be_read = bytes_per_row * rows;
double megabytes_to_be_read = (double) bytes_to_be_read / 1e6;

long getDataFromRows(int fd, unsigned char *table_data, long chunksize, long bytes_to_read) {
  
  unsigned char* curr_pos;
  long chunk = 0;
  for (; chunk < bytes_to_read/chunksize ; ++chunk) {
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
            //break;
        }
  } 
  
  // Read last part of data if needed
  /* if (chunksize * chunk < bytes_to_read) {
    curr_pos = table_data + chunksize * chunk;
    read(fd, curr_pos, bytes_to_read - chunksize * chunk);
  }  */
 return chunk;
}

int main(int argc, char **argv) {

  const char *filename =
      "/home/pp/mwa_data/G0057_1368033096_15:43:38.82_+09:29:16.30_ch109-132_0001.fits"; 

  for (long chunksize = 4096 * 1024 ; chunksize < 10000000000 ; chunksize *= 2) {
    if (chunksize > bytes_to_be_read)
      break;
    //int fd = open(filename, O_RDONLY);
    int fd = open(filename, O_RDONLY | O_DIRECT);

    if (fd == -1) {
        perror("open");
        return -1;
    }

  char smallbuffer[80];

  /* printf("Test!\n");
  for (int i = 0; i < 5; ++i) {
    printf("size 2880 block no. %d\n", i);
    for (int j = 0; j < 36; ++j) {
      int br = read(fd, smallbuffer, 80);
      if (br == -1) {
            perror("read");
            close(fd);
            exit(-1);
        }
      printf("%s\n", smallbuffer);
    }
  }
  printf("Test!\n");
  return 0; */

  /* unsigned char *table_full =
      (unsigned char *)calloc(bytes_to_be_read, sizeof(unsigned char)); */
  unsigned char *table_full;
    if (posix_memalign((void **)&table_full, 4096, bytes_to_be_read) != 0) {
        perror("posix_memalign");
        close(fd);
        return -1;
    }


  auto start_time = std::chrono::high_resolution_clock::now();

  long chunks = getDataFromRows(fd, table_full, chunksize, bytes_to_be_read);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();
  long megabytes_read = (double) (chunks * chunksize )/1e6;
  std::cout << "read " << chunks << " chunks with chunksize = " << chunksize << ", time: " << (double)duration_us / 1e6
            << " seconds" << ", Read speed (MB/s): " << megabytes_read/(double) duration_us * 1e6 << std::endl;

  free(table_full);
  close(fd);
  }
  
  return 0;
  
}