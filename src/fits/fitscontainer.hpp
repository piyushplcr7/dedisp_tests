#ifndef FITSCONTAINERHPP
#define FITSCONTAINERHPP

#include "fits.hpp"
#include <vector>
#include <string>
#include <memory>
#include <iostream>

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

    int world_rank_;
    int world_size_;
    
    //std::unique_ptr<float[]> fits_data_buf_;
    //std::unique_ptr<float[]> assembledDataBuffer; 
    //std::unique_ptr<unsigned char, void (*)(unsigned char*)> aligned_buf_{
    //    nullptr,
    //    [](unsigned char* x){ ::operator delete(x, std::align_val_t(ALIGNMENT)); }
    //};


public:
    float* fits_data_buf_;
    float* assembledDataBuffer_;
    unsigned char* aligned_buf_;
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
    //void assembleAllTimes();
    void assembleAllTimesTest();

    /*
    * This function copies the channel chunk denoted by start and end chan
    * into the contiguous buffer. This is done for the Fits object at ith 
    * position
    */
    void packChannelChunk(int i, float* contiguousData, int start_chan, int end_chan);

    /*
    * Destructor 
    */
    ~fitsLoader() {std::cout << "fitsLoader destructor called on " << world_rank_ << std::endl;}

};

#endif