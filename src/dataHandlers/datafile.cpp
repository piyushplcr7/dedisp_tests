#include "datafile.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <iostream>
#include <string>
#include <chrono>

long dataFile::getDataFromRows(int fd, unsigned char* buf, long chunksize,
                               long bytes_to_read, int fd_nodirect) {
    unsigned char* curr_pos;
    long chunk = 0;

    for (; chunk < bytes_to_read / chunksize; ++chunk) {
        curr_pos = buf + chunksize * chunk;
        ssize_t bytes_read = read(fd, curr_pos, chunksize);
        if (bytes_read == -1) { perror("read"); close(fd); exit(-1); }
        if (bytes_read == 0) { std::cerr << "Reached end of file prematurely" << std::endl; break; }
        if (bytes_read != chunksize) {
            std::cout << "read less than chunksize: " << bytes_read << std::endl;
        }
    }

    if (chunksize * chunk < bytes_to_read) {
        curr_pos = buf + chunksize * chunk;
        ssize_t bytes_read = pread(fd_nodirect, curr_pos,
                                   bytes_to_read - chunksize * chunk,
                                   chunksize * chunk);
        if (bytes_read == -1) { perror("pread"); close(fd); close(fd_nodirect); exit(-1); }
        if (bytes_read == 0) {
            std::cerr << "Reached end of file prematurely while reading last part" << std::endl;
        }
    }

    return chunk;
}

void dataFile::extractDataDirect(unsigned char* alignedBuf, size_t chunksize) {
    aligned_filesize_buffer_ = alignedBuf;

    int fd = open(filename_, O_RDONLY | O_DIRECT);
    if (fd == -1) { perror("open"); exit(-1); }

    int fd_nodirect = open(filename_, O_RDONLY);
    if (fd_nodirect == -1) { perror("open"); close(fd); exit(-1); }

#ifdef TESTDEDISP_DEBUG
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    long chunks = getDataFromRows(fd, aligned_filesize_buffer_, chunksize, file_size_, fd_nodirect);

#ifdef TESTDEDISP_DEBUG
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time).count();
    std::cout << "read " << chunks << " chunks (chunksize=" << chunksize
              << ") from " << std::string(filename_)
              << ", time: " << (double)duration_us / 1e6 << " s"
              << ", speed: " << (double)file_size_ / 1e6 / ((double)duration_us / 1e6)
              << " MB/s" << std::endl;
#endif

    close(fd);
    close(fd_nodirect);
}
