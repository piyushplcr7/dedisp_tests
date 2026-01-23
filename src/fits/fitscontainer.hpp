#ifndef FITSCONTAINERHPP
#define FITSCONTAINERHPP

#include "fits.hpp"
#include <vector>
#include <string>
#include <memory>
#include <iostream>
#include "matrix_view.hpp"

#define ALIGNMENT 4096

/*
* This class is a distributed storage of Fits data. 
*/
class fitsLoader {

private:
    std::vector<Fits> listFits_;
    int numLocFits_;
    int numGlobFits_;
    int nchans_;

    int start_chan_;
    int end_chan_;

    int world_rank_;
    int world_size_;

    int channelChunkSize_;
    size_t contiguousChunkLen_;
    size_t assembledDataLen_;
    
    std::unique_ptr<float[]> assembledDataBuffer_; 
    matrixView<float> assembledData_;

public:
    /*
    * Constructor for the distributed Fits data storage. The input is the list
    * of all the fits filenames that are logically a part of this distributed
    * storage
    */
    fitsLoader(std::vector<std::string>& listFits, int world_rank, int world_size);

    /*
    * This function goes through the local array of Fits objects sequentially and 
    * extracts and reduces the time series data
    */
    void ldSeq();

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
    * into the contiguous buffer. This is done for the Fits object at ith 
    * position
    */
    void packChannelChunk(int i, matrixView<float> contiguousDataView, int start_chan, int end_chan);

    matrixView<float> getAssembledData() { return assembledData_; }

    int startChan() { return start_chan_; }
    int endChan() { return end_chan_; }

    /*
    * Destructor 
    */
    ~fitsLoader() {}

};

#endif