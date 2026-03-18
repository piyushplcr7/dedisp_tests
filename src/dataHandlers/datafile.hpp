#ifndef DATAFILEH
#define DATAFILEH

#include <vector>
#include <iostream>

// Longest chunksize 2147479552 for direct read (linux read documentation)
#define MAX_CHUNKSIZE 2147479552
#define HALF_MAX_CHUNKSIZE 1073741824

class dataFile {
protected:
    const char* filename_;

    // Metadata strings
    char telescope_name_[40]  = {};   // derived from telescope_id
    char instrument_[100]     = {};   // derived from machine_id
    char object_name_[100]    = {};   // source_name
    char ra_[50]              = {};   // "HH:MM:SS.SSSSSS"
    char dec_[50]             = {};   // "±DD:MM:SS.SSSSSS"
    char rastring_[50]              = {};   
    char decstring_[50]             = {};
    char observer_[100]       = {};   // not in filterbank header — "Unknown"
    char projid_[100]         = {};   // not in filterbank header — "Unknown"
    char dateobs_[100]        = {};   // derived from tstart
    char ephem_[10]           = "DE405";
    char obs_[3]              = {};   // TEMPO observatory code
    char outscope_[40]        = {};   // TEMPO scope name (internal)

    // Timing / frequency
    double tbin_              = 0.0;  // tsamp (seconds)
    double epoch_             = 0.0;  // tstart (MJD)
    double fch1_              = 0.0;  // centre frequency of channel 0 (MHz)
    double foff_              = 0.0;  // channel step (MHz, negative for descending)
    double src_raj_           = 0.0;  // SIGPROC: HHMMSS.SS
    double src_dej_           = 0.0;  // SIGPROC: ±DDMMSS.SS

    // Dimensions
    int    nchans_read_       = 0;
    int    nbin_              = 1;    // always 1 (search-mode data)
    int    npol_              = 1;    // nif from header
    int    nbit_              = 8;
    int    telescope_id_      = 0;
    int    machine_id_        = 0;

    // File layout
    size_t file_size_         = 0;
    size_t file_size_aligned_ = 0;   // aligned to 4096

    // Channel frequencies in ascending order
    std::vector<double> freqs_;

    // Raw aligned IO buffer — set by extractDataDirect, used by reduceData
    unsigned char* aligned_filesize_buffer_ = nullptr;

    long getDataFromRows(int fd, unsigned char* buf, long chunksize, long bytes_to_read, int fd_nodirect);

public:
    // Virtual functions
    virtual ~dataFile() = default;
    virtual size_t dimTime(unsigned int downsamp = 1)      const = 0;
    virtual void   extractDataDirect(unsigned char* alignedBuf, size_t chunksize = HALF_MAX_CHUNKSIZE);
    virtual void   reduceData(unsigned char* buf, int out_bits=32, int poln = 0, unsigned int downsamp = 1) = 0;
    virtual void   printInfo() const noexcept = 0;
    virtual bool   checkDownsamp(unsigned int downsamp) const = 0;
    
    // Common functions
    const char* getFilename() {return filename_;}
    
    size_t getNumElements(unsigned int downsamp=1) const {
      return dimTime(downsamp) * nchans_read_;
    }

    const std::vector<double>& freqs() const noexcept { return freqs_; }

    double      sampletime(unsigned int downsamp=1)   const noexcept {return tbin_ * downsamp; }

    double      ddf()          const noexcept {
      return (freqs_[0] - freqs_[nchans_read_ - 1]) /(nchans_read_ - 1);
    }

    double      bw()           const noexcept {
      return -ddf() * nchans_read_;
    }

    double      f0()           const noexcept {return freqs_.back(); }
    const char* telescope()    const noexcept { return telescope_name_; }
    const char* instrument()   const noexcept { return instrument_; }
    const char* objectName()   const noexcept { return object_name_; }
    const char* rightAscension() const noexcept { return ra_; }
    char*       rightAscension()              { return ra_; }
    const char* declination()   const noexcept { return dec_; }
    char*       declination()                  { return dec_; }
    const char* observer()      const noexcept { return observer_; }
    const char* projid()        const noexcept { return projid_; }
    double      epoch()         const noexcept { return epoch_; }
    const char* dateobs()       const noexcept { return dateobs_; }
    const char* obs()           const noexcept { return obs_; }
    char*       obs()                          { return obs_; }
    const char* ephem()         const noexcept { return ephem_; }
    char*       ephem()                        { return ephem_; }
    size_t      fileSizeAligned() const noexcept { return file_size_aligned_; }
    size_t      fileSize()     const { return file_size_; }
    int         nchan()        const noexcept { return nchans_read_; }      // NCHAN
    double      f_lo()         const { return freqs_.front(); }
    double      f_hi()         const { return freqs_.back();  }
};

#endif