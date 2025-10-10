#ifndef FITSHPP
#define FITSHPP

#include<vector>
#include<string>
#include <fitsio.h>
#include <byteswap.h>
#include <cstdint>
#include <mpi.h>

#define MPI_CHECK(call) do { \
  int _e = (call); \
  if (_e != MPI_SUCCESS) { \
    char es[512]; int el=0; MPI_Error_string(_e, es, &el); \
    fprintf(stderr, "MPI error %s:%d: %.*s\n", __FILE__, __LINE__, el, es); \
    MPI_Abort(MPI_COMM_WORLD, _e); \
  } \
} while (0)

inline void swap_endian_3floats(float &f1, float &f2, float &f3) {
  uint32_t *p1 = reinterpret_cast<uint32_t *>(&f1);
  uint32_t *p2 = reinterpret_cast<uint32_t *>(&f2);
  uint32_t *p3 = reinterpret_cast<uint32_t *>(&f3);

  *p1 = bswap_32(*p1);
  *p2 = bswap_32(*p2);
  *p3 = bswap_32(*p3);
}

template <typename T> MPI_Datatype mpi_type();

template <> inline MPI_Datatype mpi_type<float>()         { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_type<double>()        { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_type<unsigned char>() { return MPI_UNSIGNED_CHAR; }
template <> inline MPI_Datatype mpi_type<int>()           { return MPI_INT; }


class Fits {
  public:
    // Constructor with the filename
    Fits(const char* filename);

    // ----- Getters (read-only API) -----
    [[nodiscard]] int                    num_hdus()        const noexcept { return num_hdus_; }

    [[nodiscard]] const char* telescope()    const noexcept { return telescope_name_; } // TELESCOP
    [[nodiscard]] const char* instrument()   const noexcept { return instrument_; }     // BACKEND
    [[nodiscard]] const char* source_name()  const noexcept { return object_name_; }    // SRC_NAME
    [[nodiscard]] const char* ra()           const noexcept { return ra_; }             // RA
    [[nodiscard]] const char* dec()          const noexcept { return dec_; }            // DEC
    [[nodiscard]] const char* observer()     const noexcept { return observer_; }       // OBSERVER
    [[nodiscard]] const char* projid()       const noexcept { return projid_; }         // PROJID
    [[nodiscard]] const char* dateobs()      const noexcept { return dateobs_; }        // DATE-OBS

    [[nodiscard]] int                    stt_imjd()        const noexcept { return stt_imjd_; }         // STT_IMJD
    [[nodiscard]] int                    stt_smjd()        const noexcept { return stt_smjd_; }         // STT_SMJD
    [[nodiscard]] double                 stt_offs()        const noexcept { return stt_offs_; }         // STT_OFFS
    [[nodiscard]] double                 epoch()           const noexcept { return epoch_; }            // computed
   
    [[nodiscard]] double                 tbin()            const noexcept { return tbin_; }             // TBIN
    [[nodiscard]] int                    nchan()           const noexcept { return nchans_read_; }      // NCHAN
    [[nodiscard]] int                    nsblk()           const noexcept { return nsblk_; }            // NSBLK
    [[nodiscard]] int                    naxis1()          const noexcept { return naxis1_; }           // NAXIS1
    [[nodiscard]] int                    nbin()            const noexcept { return nbin_; }             // NBIN
    [[nodiscard]] int                    npol()            const noexcept { return npol_; }             // NPOL
    [[nodiscard]] int                    naxis2()          const noexcept { return naxis2_; }           // NAXIS2

    [[nodiscard]] size_t                 data_byte_width() const noexcept { return data_byte_width_; }
    [[nodiscard]] size_t                 scal_offs_width() const noexcept { return scal_offs_width_; }
    [[nodiscard]] size_t                 file_size_aligned() const noexcept { return file_size_aligned_; }

    // Channel frequencies column (DAT_FREQ)
    [[nodiscard]] const std::vector<double>& freqs() const noexcept { return freqs_; }
    [[nodiscard]] bool                   has_freqs()       const noexcept { return !freqs_.empty(); }
    [[nodiscard]] double                 f_lo()            const noexcept { return freqs_.empty() ? 0.0 : freqs_.front(); }
    [[nodiscard]] double                 f_hi()            const noexcept { return freqs_.empty() ? 0.0 : freqs_.back();  }

    long getDataFromRows(int fd, unsigned char *table_data, long chunksize, long bytes_to_read, int fd_nodirect);
    
    void reduceBinaryTable(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width);

    void reduceBinaryTableChunk(unsigned char *full_binary_table, float *data, int poln,
                       int naxis1, int naxis2, int nsblk, int nchans, int npol,
                       size_t data_offset_from_start, size_t data_cols_offset,
                       size_t scal_offs_width, int chan_start, int chan_end);
    
    // Extracting all the data from the fits file using the provided aligned_filesize_buffer
    void extractDataDirect(size_t chunksize=1073741824);

    // Reduce the data into the data buffer. The first dimension in the reduced data is channels
    void reduceData(int poln=0);

    void setAlignedFileSizeBuffer(unsigned char* aligned_filesize_buffer) { aligned_filesize_buffer_ = aligned_filesize_buffer; }
    void setDataBuffer(float* data) { data_ = data; }

    float* data() { return data_; }
    unsigned char* mpiRawdataBuffer() { return mpi_rawdata_buffer_;}

    int nchanLocal() {return nchan_local_;}
    int chanStartLocal() { return chan_start_local_;}

    template <typename DATATYPE>
    void readColsFromTableDistributed(int rank, int size, int rows_table, int cols_table, int start_col, int cols_to_read, DATATYPE* buf);

    template <typename T>
    std::pair<T,T> getMySize(int rank, int size, T total_size);

    void computeMPIChunkInfo(int rank, int size);

    void buildFileTypeRawdataSinglePoln(int rank, int size);

    void readDataMPI(int rank, int size, int poln);

    void setVerbosity(int verbosity) {verbose = verbosity;}

    void reduceDataMPI(int rank, int size, float* data_buffer);

    ~Fits() {
      if (mpi_rawdata_buffer_)         free(mpi_rawdata_buffer_);
      if (mpi_reduced_data_buffer_) free(mpi_reduced_data_buffer_);
      if (mpi_chan_weights_buffer_) free(mpi_chan_weights_buffer_);
      if (mpi_chan_offsets_buffer_) free(mpi_chan_offsets_buffer_);
      if (mpi_chan_scales_buffer_)  free(mpi_chan_scales_buffer_);
    };
  
  private:
    const char* filename_;
    int         num_hdus_          = 0;

    // Strings stored in char arrays (fixed-size buffers)
    char telescope_name_[100] = {};  // TELESCOP
    char instrument_[100]     = {};  // BACKEND
    char object_name_[100]    = {};  // SRC_NAME
    char ra_[100]             = {};  // RA
    char dec_[100]            = {};  // DEC
    char observer_[100]       = {};  // OBSERVER
    char projid_[100]         = {};  // PROJID
    char dateobs_[100]        = {};  // DATE-OBS

    int         stt_imjd_          = 0;     // STT_IMJD
    int         stt_smjd_          = 0;     // STT_SMJD
    double      stt_offs_          = 0.0;   // STT_OFFS
    double      epoch_             = 0.0;   // computed: IMJD + (SMJD + OFFS)/86400

    // SUBINT HDU keys
    double      tbin_              = 0.0;   // TBIN
    int         nchans_read_       = 0;     // NCHAN
    int         nsblk_             = 0;     // NSBLK
    int         naxis1_            = 0;     // NAXIS1
    int         nbin_              = 0;     // NBIN
    int         npol_              = 0;     // NPOL
    int         naxis2_            = 0;     // NAXIS2
    int         verbose         = 0;

    size_t      data_byte_width_   = 0;     // nbin * nchan * npol * nsblk (bytes if 1 byte/sample)
    size_t      scal_offs_width_   = 0;     // nchan * npol
    size_t fits_header_bytesize_   = 0;     // Offset for data in the binary table (bytes)
    size_t data_block_bytesize_    = 0;     // Size of data (bytes)
    size_t file_size_              = 0;     // File size in bytes
    size_t file_size_aligned_      = 0;     // File size aligned to 4096 bytes
    size_t start_col_bytes_        = 0;     // Initial offset to reach data (bytes)

    std::vector<double> freqs_;             // DAT_FREQ column (size = nchan)

    float* data_                            = nullptr; 
    unsigned char* aligned_filesize_buffer_ = nullptr;

    // MPI related buffers
    unsigned char* mpi_rawdata_buffer_      = nullptr;
    float* mpi_reduced_data_buffer_         = nullptr;
    float* mpi_chan_weights_buffer_         = nullptr;
    float* mpi_chan_offsets_buffer_         = nullptr;
    float* mpi_chan_scales_buffer_          = nullptr;
    
    bool mpi_buffers_created_              = false;

    MPI_Datatype filetype_rawdata_;
    int nchan_local_;
    int chan_start_local_;
};

#endif