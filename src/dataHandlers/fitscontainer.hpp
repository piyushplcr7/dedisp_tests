#ifndef FITSCONTAINERHPP
#define FITSCONTAINERHPP

#include "datafile.hpp"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include "matrix_view.hpp"

// Runtime O_DIRECT alignment (page size, see datafile.hpp). Kept as a macro
// for the existing call sites that build std::align_val_t(ALIGNMENT).
#define ALIGNMENT directIOAlignment()

/*
* This class is a distributed storage of Fits data.
*/
class dataLoader {

private:
    std::vector<std::unique_ptr<dataFile>> listFiles_;
    int numLocFits_;
    int numGlobFits_;

    // Number of channels owned by the container after its usage is determined
    int nchans_ = 0;

    int start_chan_;
    int end_chan_;

    int world_rank_;
    int world_size_;

    int downsamp_;

    double maxvoverc_= -1;
    double minvoverc_ = 1;
    double avgvoverc_ = 0;
    double blotoa_ = 0.0;

    size_t contiguousChunkLen_ = 0;
    size_t assembledDataLen_ = 0;

    size_t nsampsLocal_ = 0;
    size_t nsampsGlobal_ = 0;

    // Header-only handle to the very first global file (listFitsNames[0]),
    // opened on every rank so barycenter() can compute the global resample
    // map deterministically without MPI communication.
    std::unique_ptr<dataFile> firstFile_;

    std::unique_ptr<float[]> assembledDataBuffer_;

    std::vector<int> inForOut_;
    std::vector<int> insertPositions_;
    std::vector<int> diffbins_;
    int nbits_;

public:
    const int nbits() const { return nbits_; }
    const std::vector<int>& getInsertPositions() const { return insertPositions_; }
    const std::vector<int>& getInForOut() const { return inForOut_; }
    const std::vector<int>& getDiffbins() const { return diffbins_; }
    const int getMPIRank() const {return world_rank_;}
    const int getMPISize() const {return world_size_;}

    /*
    * Constructor for the distributed Fits data storage. The input is the list
    * of all the fits filenames that are logically a part of this distributed
    * storage
    */
    dataLoader(std::vector<std::string>& listFits, int world_rank, int world_size, int downsamp, int nbits);

    void barycenter();

    /*
    * This function goes through the local array of Fits objects sequentially and
    * extracts and reduces the time series data
    */
    void ldSeq(size_t chunksize, int poln);

    void parLoad();

    double avgvoverc() const { return avgvoverc_; }
    double blotoa()    const { return blotoa_; }

    /*
    * In a multi-node setting, where each node holds a contiguous time chunk of
    * the entire time series, this function assembles all the time points on
    * each node for the local channel chunk. So nodes go from having all channels
    * and a time chunk to all times and a channel chunk.
    */
    void assembleAllTimes();

    void assembleAllTimesAsync();

    /*
    * This function copies the channel chunk denoted by start and end chan
    * into the contiguous buffer.
    */
    void packChannelChunk(matrixView<float> dataView, matrixView<float> contiguousDataView, int start_chan, int end_chan);

    std::unique_ptr<float[]>& getAssembledDataBfr() { return assembledDataBuffer_; }

    /*
    * Returns a byte pointer to the start of the i-th segment in the assembled
    * data buffer. Each segment corresponds to one file's reduced timeseries,
    * laid out contiguously in the order they appear in listFiles_.
    */
    unsigned char* getSegmentPtr(int i) const;

    int numLocFits() const { return numLocFits_; }

    int numGlobFits() const {return numGlobFits_; }

    size_t nsampsPerSegment() const { return listFiles_[0]->dimTime(downsamp_); }

    /*
    * Returns true iff all local files have the same dimTime(downsamp_).
    * Used to validate that per-file segments in the assembled buffer are
    * equal length.
    */
    bool segmentsEqualLength() const;

    int startChan() const { return start_chan_; }
    int endChan() const { return end_chan_; }

    void allocContiguousAssemblyBuf();

    void reduceInAssemblyBuf();

    double sampletime() const { return listFiles_[0]->sampletime(downsamp_); }

    size_t nsampsLocal() const { return nsampsLocal_; }

    size_t nsampsGlobal() const { return nsampsGlobal_; }

    double f0() const { return listFiles_[0]->f0(); }

    int nchansGlobal() const { return nchans_; }

    int nchansLocal() const { return end_chan_ - start_chan_; }

    const std::vector<std::unique_ptr<dataFile>>& getFileVector() const { return listFiles_; }

    double ddf() const { return listFiles_[0]->ddf(); }

    double bw() const { return listFiles_[0]->bw(); }

    ~dataLoader() {}
};

#endif
