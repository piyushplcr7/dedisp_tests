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
#include <mpi.h>
#include <tuple>

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

  // Get number of HDUs
  fits_get_num_hdus(ffptr, &num_hdus_, &status);
  if (verbose)
    printf("No. of HDUs = %d\n", num_hdus_);

  char comment[80];
  fits_read_key(ffptr, TSTRING, "TELESCOP", telescope_name_, comment, &status);
  if (verbose)
    printf("telescope name = %s \n", telescope_name_);

  fits_read_key(ffptr, TSTRING, "BACKEND", instrument_, comment, &status);
  if (verbose)
    printf("instrument name = %s \n", instrument_);

  fits_read_key(ffptr, TSTRING, "SRC_NAME", object_name_, comment, &status);
  if (verbose)
    printf("object name = %s \n", object_name_);

  fits_read_key(ffptr, TSTRING, "RA", ra_, comment, &status);
  if (verbose)
    printf("right_ascension = %s \n", ra_);

  fits_read_key(ffptr, TSTRING, "DEC", dec_, comment, &status);
  if (verbose)
    printf("declination = %s \n", dec_ + 1);

  fits_read_key(ffptr, TSTRING, "OBSERVER", observer_, comment, &status);
  if (verbose)
    printf("observer = %s \n", observer_);

  /* Read and print the integer keyword STT_IMJD */
  if (fits_read_key(ffptr, TINT, "STT_IMJD", &stt_imjd_, comment, &status)) {
      fits_report_error(stderr, status);
      //return(status);
  }
  if (verbose)
    printf("STT_IMJD = %d   (%s)\n", stt_imjd_, comment);

  /* Read and print the integer keyword STT_SMJD */
  if (fits_read_key(ffptr, TINT, "STT_SMJD", &stt_smjd_, comment, &status)) {
      fits_report_error(stderr, status);
      //return(status);
  }
  if (verbose)
    printf("STT_SMJD = %d   (%s)\n", stt_smjd_, comment);

  /* Read and print the double keyword STT_OFFS */
  if (fits_read_key(ffptr, TDOUBLE, "STT_OFFS", &stt_offs_, comment, &status)) {
      fits_report_error(stderr, status);
      //return(status);
  }
  if (verbose)
    printf("STT_OFFS = %.15f   (%s)\n", stt_offs_, comment);

  /* Compute the final epoch: epoch = STT_IMJD + (STT_SMJD + STT_OFFS)/86400 */
  epoch_ = stt_imjd_ + ((stt_smjd_ + stt_offs_) / 86400.0);
  if (verbose)
    printf("Computed epoch = %.15f\n", epoch_);

  /* Read and print the integer keyword STT_SMJD */
  if (fits_read_key(ffptr, TSTRING, "PROJID", projid_, comment, &status)) {
      fits_report_error(stderr, status);
      //return(status);
  }
  if (verbose)
    printf("projid = %s   (%s)\n", projid_, comment);

/* Read and print the integer keyword STT_SMJD */
  if (fits_read_key(ffptr, TSTRING, "DATE-OBS", dateobs_, comment, &status)) {
      fits_report_error(stderr, status);
      //return(status);
  }
  if (verbose)
    printf("dateobs = %s   (%s)\n", dateobs_, comment);

  // Moving to the relevant hdu
  fits_movnam_hdu(ffptr, BINARY_TBL, "SUBINT", 0, &status);

  fits_read_key(ffptr, TDOUBLE, "TBIN", &tbin_, comment, &status);
  if (verbose)
    printf("read tbin = %f\n", tbin_);

  fits_read_key(ffptr, TINT, "NCHAN", &nchans_read_, comment, &status);
  if (verbose)
    printf("Nchans read = %d\n", nchans_read_);

  fits_read_key(ffptr, TINT, "NSBLK", &nsblk_, comment, &status);
  if (verbose)
    printf("nsblk = %d\n", nsblk_);

  fits_read_key(ffptr, TINT, "NAXIS1", &naxis1_, comment, &status);
  if (verbose)
    printf("naxis1 = %d\n", naxis1_);

  fits_read_key(ffptr, TINT, "NBIN", &nbin_, comment, &status);
  if (verbose)
    printf("nbin = %d\n", nbin_);

  fits_read_key(ffptr, TINT, "NPOL", &npol_, comment, &status);
  if (verbose)
    printf("npol = %d\n", npol_);

  data_byte_width_ = (size_t)nbin_ * nchans_read_ * npol_ * nsblk_;
  if (verbose)
    printf("data byte width = %ld\n", data_byte_width_);
  scal_offs_width_ = (size_t)nchans_read_ * npol_;
  if (verbose)
    printf("offs, scal width = %ld\n", scal_offs_width_);

  fits_read_key(ffptr, TINT, "NAXIS2", &naxis2_, comment, &status);
  if (verbose)
    printf("naxis2 = %d\n", naxis2_);

  int colnum = -1;
  if (verbose)
    fits_get_colnum(ffptr, 0, "DAT_FREQ", &colnum, &status);

  freqs_.assign(nchans_read_, 0.0);
  if (status == COL_NOT_FOUND) {
    printf("Warning!:  Can't find the channel freq column!\n");
    status = 0; // Reset status
  } else {
    int anynull = 0;
    fits_read_col(ffptr, TDOUBLE, colnum, 1L, 1L, nchans_read_, 0, freqs_.data(),
                  &anynull, &status);
  }

  if (verbose)
    printf("lo, hi freq = %.15f, %.15f\n", freqs_[0], freqs_[nchans_read_ - 1]);

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
  fits_header_bytesize_ = i * 2880;
  data_block_bytesize_ = (size_t)naxis1_ * naxis2_;
  file_size_ = fits_header_bytesize_+ data_block_bytesize_;
  file_size_aligned_ = ((file_size_ + 4095) / 4096) * 4096;
  start_col_bytes_ =
      (size_t)naxis1_ - 4 * nchans_read_ - 2 * 4 * scal_offs_width_ - data_byte_width_;

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

  auto start_time = std::chrono::high_resolution_clock::now();
  long chunks =
      getDataFromRows(fd, aligned_filesize_buffer_, chunksize, file_size_, fd_nodirect);

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

  close(fd);
  close(fd_nodirect);
}

void Fits::reduceData(int poln) {
  if (aligned_filesize_buffer_ == nullptr) {
    std::cerr << "Error: aligned_filesize_buffer_ is not set. Use setAlignedFileSizeBuffer() before calling reduceData()." << std::endl;
    exit(-1);
  }
  if (data_ == nullptr) {
    std::cerr << "Error: data_ is not set. Use setDataBuffer() before calling reduceData()." << std::endl;
    exit(-1);
  }
  // Reducing the binary table to get the relevant data
  auto start_time = std::chrono::high_resolution_clock::now();
  reduceBinaryTable(aligned_filesize_buffer_, data_, poln, naxis1_, naxis2_,
                    nsblk_, nchans_read_, npol_, fits_header_bytesize_, start_col_bytes_,
                    scal_offs_width_);

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                    end_time - start_time)
                    .count();

  std::cout << "reduction finished, time: " << (double)duration_us / 1e6
            << std::endl;
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
                       size_t header_byte_size, size_t data_cols_offset,
                       size_t scal_offs_width) {
  // Skipping the header
  unsigned char *bin_table_start = full_binary_table + header_byte_size;
  size_t nchans_poln = nchans * poln;

// Going over all the rows (subints) of the binary table
#pragma omp parallel
  {
    //int num_threads = omp_get_num_threads();
    // std::cout << "omp_num_threads = " << num_threads << std::endl;
    //int thread_id = omp_get_thread_num();

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

          // No byteswapping
          data[nsblk_nchans_subint_nchans_spectra + chan] =
              ((float)rawdata[nchans_npol_spectra_nchans_poln + chan] *
                   data_scl[nchans_poln + chan] +
               data_offs[nchans_poln + chan]) *
              data_wts[chan];
        }
      }
    }
  }
}

void Fits::reduceBinaryTableChunk(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width, int chan_start, int chan_end) {
  // Skipping the header
  unsigned char *bin_table_start = full_binary_table + data_offset_from_start;
  size_t nchans_poln = nchans * poln;

// Going over all the rows (subints) of the binary table
#pragma omp parallel
  {
    //int num_threads = omp_get_num_threads();
    // std::cout << "omp_num_threads = " << num_threads << std::endl;
    //int thread_id = omp_get_thread_num();

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

          // No byteswapping
          data[nsblk_nchans_subint_nchans_spectra + chan] =
              ((float)rawdata[nchans_npol_spectra_nchans_poln + chan] *
                   data_scl[nchans_poln + chan] +
               data_offs[nchans_poln + chan]) *
              data_wts[chan];
        }
      }
    }
  }
}

template <typename T>
std::pair<T,T> Fits::getMySize(int rank, int size, T total_size) {
  T size_per_rank = (total_size + size - 1) / size; // Ceiling division
  T start_index = rank * size_per_rank;
  T end_index = std::min(start_index + size_per_rank, total_size);
  return std::make_pair(start_index, end_index - start_index);
}

void Fits::computeMPIChunkInfo(int rank, int size) {
  std::tie(chan_start_local_, nchan_local_) = getMySize(rank,size,nchans_read_);
}

void Fits::buildFileTypeRawdataSinglePoln(int rank, int size) {
  // Data layout
  // [columns of other data, spectra1( pol0(ch0, ch1,...), pol1,... ), spectra2,... ]
  MPI_Datatype etype = mpi_type<unsigned char>();
  
  computeMPIChunkInfo(rank,size);

  MPI_Aint intra_row_stride = nchans_read_ * npol_;

  // Defining the structure of elements of interest in one single row
  MPI_Datatype row_slice;
  MPI_Type_create_hvector(nsblk_, // No. of elements in one row of subint
                          nchan_local_,   // local channels num of interest for the process
                          intra_row_stride,   // stride between chunks in one row
                          etype, &row_slice);
  MPI_Type_commit(&row_slice);

  // Repeating the structure in a single row to span the entire fits
  // The stride between these structures across rows is just naxis1_ (row size in bytes)
  MPI_Datatype row_type;
  MPI_Type_create_resized(row_slice, 0, naxis1_, &row_type);
  MPI_Type_commit(&row_type);
  MPI_Type_free(&row_slice);

  // Repeat for all rows?
  MPI_Type_contiguous(naxis2_, row_type, &filetype_rawdata_);
  MPI_Type_commit(&filetype_rawdata_);
  MPI_Type_free(&row_type);
}

void Fits::readDataMPI(int rank, int size, int poln) {
  double t0, t1;

  t0 = MPI_Wtime();
  // Building filetype for the rawdata
  buildFileTypeRawdataSinglePoln(rank,size);

  // Open file with helpful ROMIO hints (collective buffering)
  MPI_Info info; MPI_Info_create(&info);
  MPI_Info_set(info, "romio_cb_read", "enable");
  // Choose a sensible number of aggregators (e.g., ~IO servers or 1–4 per node)
  MPI_Info_set(info, "cb_nodes", "64");                    // tune for your system
  MPI_Info_set(info, "cb_buffer_size", "16777216");        // 16 MiB (multiple of stripe size)
  MPI_Info_set(info, "romio_ds_read", "disable");          // prefer two-phase over data-sieving
  MPI_Info_set(info, "romio_no_indep_rw", "true");         // force collective path

  MPI_File fh;
  MPI_CHECK(MPI_File_open(MPI_COMM_WORLD, filename_, MPI_MODE_RDONLY, info, &fh));
  MPI_Info_free(&info);

  MPI_CHECK(MPI_File_set_errhandler(fh, MPI_ERRORS_RETURN));

  // Offset for the unsigned char raw data!
  size_t rawdata_start_col_bytes = (size_t)naxis1_ - data_byte_width_;

  // Offset for the first element of the polarization and the channel block we want to read
  MPI_Offset rawdata_offset_local = fits_header_bytesize_ + rawdata_start_col_bytes + (MPI_Offset)nchans_read_ * poln + (MPI_Offset)chan_start_local_;
  
  // Calculate local data size (just count, not including the data type size)
  MPI_Count local_rawdata_size = (MPI_Count)naxis2_ * nsblk_ * nchan_local_;

  // Allocate the buffer before reading
  mpi_rawdata_buffer_ = (unsigned char*)malloc(local_rawdata_size * sizeof(unsigned char));
  if (!mpi_rawdata_buffer_) {
    perror("malloc"); 
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_NO_MEM); 
  }
  
  MPI_CHECK(MPI_File_set_view(fh, rawdata_offset_local, MPI_UNSIGNED_CHAR, filetype_rawdata_, "native", MPI_INFO_NULL));
  
  MPI_Status st;
  MPI_Count rem = local_rawdata_size;
  unsigned char* p = mpi_rawdata_buffer_;
  while (rem > 0) {
    int chunk = (rem > INT_MAX) ? INT_MAX : (int)rem;
    MPI_CHECK(MPI_File_read_all(fh, p, chunk, MPI_UNSIGNED_CHAR, &st));
    p   += chunk;
    rem -= chunk;

    int got = 0; 
    MPI_Get_count(&st, MPI_UNSIGNED_CHAR, &got);

    if (got != chunk) { 
      fprintf(stderr, "Short read: got %d of %d elems\n", got, chunk); 
      MPI_Abort(MPI_COMM_WORLD, MPI_ERR_IO); 
    }
  }
  
  // The shorter reads below will use the same chunksize, maybe something to change?

  // Reading data wts
  // Build the filetype
  MPI_Datatype mpi_datawt_type;
  int global_size[2] = {naxis2_, naxis1_/sizeof(float)};
  int subsize[2] = {naxis2_, nchan_local_};
  int start[2] = {0, chan_start_local_};
  MPI_CHECK(MPI_Type_create_subarray(2, global_size, subsize, start, MPI_ORDER_C, MPI_FLOAT, &mpi_datawt_type));
  MPI_Type_commit(&mpi_datawt_type);

  // allocate buffers
  size_t wts_count = (size_t) naxis2_ * nchan_local_;
  mpi_chan_weights_buffer_ = (float*) malloc(wts_count * sizeof(float));
  mpi_chan_offsets_buffer_ = (float*) malloc(wts_count * sizeof(float));
  mpi_chan_scales_buffer_ = (float*) malloc(wts_count * sizeof(float));

  // Computing the byte offsets
  MPI_Offset data_wts_offset = fits_header_bytesize_ + start_col_bytes_ + (MPI_Offset) chan_start_local_ * sizeof(float);
  MPI_Offset data_offs_offset = fits_header_bytesize_ + start_col_bytes_ + 
                                (MPI_Offset) nchans_read_ * sizeof(float) + 
                                (MPI_Offset)poln * nchans_read_ * sizeof(float) + (MPI_Offset) chan_start_local_ * sizeof(float);  

  MPI_Offset data_scls_offset = fits_header_bytesize_ + start_col_bytes_ + 
                                (MPI_Offset) nchans_read_ * sizeof(float) + 
                                (MPI_Offset)npol_ * nchans_read_ * sizeof(float) + 
                                (MPI_Offset)poln * nchans_read_ * sizeof(float) + (MPI_Offset) chan_start_local_ * sizeof(float);  

  // Set the view to extract data wts
  MPI_CHECK(MPI_File_set_view(fh, data_wts_offset, MPI_FLOAT, mpi_datawt_type, "native", MPI_INFO_NULL));
  MPI_CHECK(MPI_File_read_all(fh, mpi_chan_weights_buffer_, wts_count, MPI_FLOAT, MPI_STATUS_IGNORE));

  // Set the view to extract data offsets
  MPI_CHECK(MPI_File_set_view(fh, data_offs_offset, MPI_FLOAT, mpi_datawt_type, "native", MPI_INFO_NULL));
  MPI_CHECK(MPI_File_read_all(fh, mpi_chan_offsets_buffer_, wts_count, MPI_FLOAT, MPI_STATUS_IGNORE));

  // Set the view to extract data scales
  MPI_CHECK(MPI_File_set_view(fh, data_scls_offset, MPI_FLOAT, mpi_datawt_type, "native", MPI_INFO_NULL));
  MPI_CHECK(MPI_File_read_all(fh, mpi_chan_scales_buffer_, wts_count, MPI_FLOAT, MPI_STATUS_IGNORE));

  MPI_Type_free(&mpi_datawt_type);
  MPI_Type_free(&filetype_rawdata_);
  MPI_File_close(&fh);

  t1 = MPI_Wtime();

  if (verbose)
    std::cout << "Process " << rank << " finished read in " << t1-t0 << " seconds" << std::endl;
}

void Fits::reduceDataMPI(int rank, int size, float* data_buffer) {
  float *subint_out_buffer, *subint_wts, *subint_scales, *subint_offsets;
  unsigned char* subint_rawdata_buffer;

  auto start_time = std::chrono::high_resolution_clock::now();  
  // Go over the subints
  #pragma omp parallel for
  for (int i = 0 ; i < naxis2_ ; ++i) {
    size_t subint_offset = (size_t)i * nsblk_ * nchan_local_;
    subint_out_buffer = data_buffer + subint_offset;
    subint_rawdata_buffer = mpi_rawdata_buffer_ + subint_offset;

    size_t chan_offset = (size_t)i * nchan_local_;
    subint_wts = mpi_chan_weights_buffer_ + chan_offset;
    subint_scales = mpi_chan_scales_buffer_ + chan_offset;
    subint_offsets = mpi_chan_offsets_buffer_ + chan_offset;

    // Byte swapping the weights, offsets and scales
    for (int chan = 0 ; chan < nchan_local_ ; ++chan) {
      swap_endian_3floats(subint_wts[chan], 
                          subint_offsets[chan], 
                          subint_scales[chan]);
    }

    for (int spectra = 0 ; spectra < nsblk_ ; ++spectra) {

      float* spectra_out_buffer = subint_out_buffer + (size_t)spectra * nchan_local_;
      unsigned char* spectra_rawdata_buffer = subint_rawdata_buffer + (size_t)spectra * nchan_local_;

      for (int chan = 0 ; chan < nchan_local_ ; ++chan) {
        spectra_out_buffer[chan] = subint_wts[chan] *
         ((float)spectra_rawdata_buffer[chan] * subint_scales[chan] + subint_offsets[chan]);

      }
    }
  }  

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_time - start_time)
                    .count();

  if (verbose)
    std::cout << "Process " << rank << " finished reduction in " << duration << " ms" << std::endl;
}

/* template <typename DATATYPE>
void Fits::readColsFromTableDistributed(int rank, int size, int rows_table, int cols_table, int start_col, int cols_to_read, DATATYPE* buf) {
  // Header offset in bytes
  const MPI_Offset header_bytes = fits_header_bytesize_;

  MPI_Datatype mpi_dtype = mpi_type<DATATYPE>();

  // Determine the column range for this rank
  const int cols_per_rank = (cols_to_read + size - 1) / size; // Ceiling division
  const int local_col_start = start_col + rank * cols_per_rank;
  int local_col_end = local_col_start + cols_per_rank;
  if (local_col_end > start_col + cols_to_read) {
    local_col_end = start_col + cols_to_read;
  }
  const int local_num_cols = local_col_end - local_col_start;

  // Throw error if no columns to read
  if (local_num_cols <= 0) {
    std::cerr << "Rank " << rank << " has no columns to read." << std::endl;
    exit(-1);
  }

  // Open file with helpful ROMIO hints (collective buffering)
  MPI_Info info; MPI_Info_create(&info);
  MPI_Info_set(info, "romio_cb_read", "enable");
  // Choose a sensible number of aggregators (e.g., ~IO servers or 1–4 per node)
  MPI_Info_set(info, "cb_nodes", "64");                    // tune for your system
  MPI_Info_set(info, "cb_buffer_size", "16777216");        // 16 MiB (multiple of stripe size)
  MPI_Info_set(info, "romio_ds_read", "disable");          // prefer two-phase over data-sieving
  MPI_Info_set(info, "romio_no_indep_rw", "true");         // force collective path

  MPI_File fh;
  int mpi_err = MPI_File_open(MPI_COMM_WORLD, filename_, MPI_MODE_RDONLY, info, &fh);
  MPI_Info_free(&info);

  // Check for errors
  if (mpi_err != MPI_SUCCESS) {
    char err_string[MPI_MAX_ERROR_STRING];
    int err_length;
    MPI_Error_string(mpi_err, err_string, &err_length);
    std::cerr << "Rank " << rank << " failed to open file: " << err_string << std::endl;
    MPI_Abort(MPI_COMM_WORLD, mpi_err);
  }

  // Build subarray datatype for reading
  MPI_Datatype filetype;
  int sizes[2]    = {rows_table, cols_table};      // global array size
  int subsizes[2] = {rows_table, local_num_cols};    // local array size
  int starts[2]   = {0, local_col_start};          // starting point of local array
  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, mpi_dtype, &filetype);
  MPI_Type_commit(&filetype);

  // Set the file view
  MPI_File_set_view(fh, fits_header_bytesize_, mpi_dtype, filetype, "native", MPI_INFO_NULL);

  // Calculate local data size
  MPI_Count local_data_size = (MPI_Count)rows_table * local_num_cols;

  // MPI Read the data collectively
  MPI_File_read_all(fh, buf, local_data_size, mpi_dtype, MPI_STATUS_IGNORE);

  MPI_Type_free(&filetype);
  MPI_File_close(&fh);
} */

