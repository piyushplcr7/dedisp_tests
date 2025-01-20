
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <chrono>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <fcntl.h>
#include <unistd.h>

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "fdd_gpu.h"
#include "fitsio.h"

#include <Plan.hpp>
#include <cuda_runtime.h>

#include "fdd/helper.h"
#define DIRECTRD

static int rows = 200;
static long bytes_per_row = 123002956;
static long bytes_to_be_read = bytes_per_row * rows;
double megabytes_to_be_read = (double) bytes_to_be_read / 1e6;

long getDataFromRows(int fd, unsigned char *table_data, long chunksize, long bytes_to_read, int fd_nodirect) {
  unsigned char* curr_pos;
  long chunk = 0;
  for ( ; chunk < bytes_to_read/chunksize; ++chunk) {
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
  
  // Read last part of data using the alternative file descriptor if needed
  if (chunksize * chunk < bytes_to_read) {
    curr_pos = table_data + chunksize * chunk;
    ssize_t bytes_read = pread(fd_nodirect, curr_pos, bytes_to_read - chunksize * chunk, chunksize * chunk);

    if (bytes_read == -1) {
      perror("pread");
      close(fd);
      close(fd_nodirect);
      exit(-1);
    }

    if (bytes_read == 0) {
      std::cerr << "Reached end of file prematurely while reading last part" << std::endl;
    }
  } 
 return chunk;
}

int main(int argc, char **argv) {
  const char *filename =
      "/home/pp/mwa_data/test.fits"; 

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

  fits_close_file(ffptr, &status);

  // Reading the fits file manually to find the byte size of HDUs, ie offset after which data begins
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
      char* tempptr = strstr(smallbuffer, "END     ");
      if (tempptr != NULL) {
        num_hdus--;
      }
    }
  }

  size_t offset_for_data = i * 2880;
  size_t data_size = (size_t) naxis1 * naxis2;
  std::cout << "i = " << i << std::endl;
  std::cout << "offset_for_data = " << offset_for_data << std::endl;
  std::cout << "naxis1 * naxis2 = " << (size_t)naxis1 * naxis2 << std::endl;
  std::cout << "data size = " << data_size << std::endl;
  size_t file_size = offset_for_data + data_size;
  //size_t file_size_aligned = (file_size/4096+1)*4096;
  size_t file_size_aligned = ( (file_size + 4095)/4096 )*4096;
  std::cout << "file size = " << file_size << ", aligned size = " << file_size_aligned << std::endl; 

  fclose(fptr);
  // Longest chunksize 2147479552 (linux read documentation) 
  long chunksize = 2147479552;

  //for (long chunksize = 4096 * 1024 ; chunksize < 10000000000 ; chunksize *= 2) {
  //for (long chunksize = 2147479552 ; chunksize < 2147479554 ; chunksize *= 2) {
    /* if (chunksize > bytes_to_be_read)
      break; */
    #ifdef DIRECTRD
    int fd = open(filename, O_RDONLY | O_DIRECT);
    int fd_nodirect = open(filename, O_RDONLY);
    #else
    int fd = open(filename, O_RDONLY);
    #endif

    if (fd == -1) {
        perror("open");
        return -1;
    }


  #ifndef DIRECTRD
  char smallbuffer[80];
  //printf("Test!\n");
  for (int i = 0; i < 5; ++i) {
    //printf("size 2880 block no. %d\n", i);
    for (int j = 0; j < 36; ++j) {
      int br = read(fd, smallbuffer, 80);
      if (br == -1) {
            perror("read");
            close(fd);
            exit(-1);
        }
      //printf("%s\n", smallbuffer);
    }
  }
  //printf("Test!\n");
  #endif

  #ifdef DIRECTRD
  unsigned char *table_full;
    if (posix_memalign((void **)&table_full, 4096, file_size_aligned) != 0) {
        perror("posix_memalign");
        close(fd);
        return -1;
    }
  #else
  unsigned char *table_full =
      (unsigned char *)calloc(bytes_to_be_read, sizeof(unsigned char));
  #endif


  auto start_time = std::chrono::high_resolution_clock::now();

  long chunks = getDataFromRows(fd, table_full, chunksize, file_size, fd_nodirect);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time)
                         .count();
  long megabytes_read = (double) (chunks * chunksize )/1e6;
  std::cout << "read " << chunks << " chunks with chunksize = " << chunksize << ", time: " << (double)duration_us / 1e6
            << " seconds" << ", Read speed (MB/s): " << megabytes_read/(double) duration_us * 1e6 << std::endl;

  close(fd);
 // }
  // Reading again to create a reference table
  
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
        close(fd);
        return EXIT_FAILURE;
    }

  unsigned char *table_full_fread = (unsigned char*) mapped;

  std::cout << "Comparing the two binary tables" << std::endl;

  for (size_t i = 0 ; i < file_size ; ++i) {
    if (table_full[i] != table_full_fread[i]) {
      std::cout << "mismatch at i = " << i << std::endl;
    }
  }

  std::cout << "perfect match!!!!!!!"  << std::endl;
  
  free(table_full);
  free(table_full_fread);

  return 0;
  
}