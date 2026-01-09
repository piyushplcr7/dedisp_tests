#include "fitscontainer.hpp"
#include <vector>
#include <iostream>
#include <memory>
#include <mpi.h>

fitsLoader::fitsLoader(std::vector<std::string>& listFitsNames, int world_rank, int world_size) { 
    std::cout << "Entered constructor" << std::endl;

    world_rank_ = world_rank;
    world_size_ = world_size;

    // Check if configuration is allowed
    if (listFitsNames.size() % world_size_ != 0) {
        std::cerr << "Number of fits files not divisible by MPI size! Aborting." << std::endl;
        exit(-1);
    }
    
    if (listFitsNames.size() == 0) {
        std::cerr << "Number of fits files = 0 is not allowed! Aborting." << std::endl;
        exit(-1);
    } 

    // If configuration is allowed, set the number of local fits
    numGlobFits_ = listFitsNames.size();
    numLocFits_ = numGlobFits_ / world_size_;

    std::cout << "numLocFits_ = " << numLocFits_ << std::endl;

    listFits_.reserve(numLocFits_);
    

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        std::cout << "Constructing fits object using file: " << listFitsNames[world_rank_ * numLocFits_ + i].c_str() << std::endl;
        listFits_.emplace_back(listFitsNames[world_rank_ * numLocFits_ + i].c_str());
        std::cout << "Loop iteration no. " << i << " finished" << std::endl;
    }

    nchans_ = listFits_[0].nchan();

    if (nchans_ % world_size_ != 0) {
        std::cerr << "Error, nchans not divisible by npes = " << world_size_ << std::endl;
        exit(-1);
    }

    std::cout << "aligned buf size = " << listFits_[0].getAlignedBufSize() << std::endl;
    
    // Allocate aligned buffer once which will be reused for all Fits
    //aligned_buf_ = static_cast<unsigned char*>(std::aligned_alloc(4096, listFits_[0].getAlignedBufSize()));
    aligned_buf_ = static_cast<unsigned char*>(std::aligned_alloc(ALIGNMENT, listFits_[0].getAlignedBufSize()));


    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFits_[i].setAlignedFileSizeBuffer(aligned_buf_);
    }

    // Allocate space for the reduced data
    fits_data_buf_ = static_cast<float*> (std::malloc(listFits_[0].getNumElements() * sizeof(float))); //std::make_unique<float[]>(listFits_[0].getNumElements());

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFits_[i].setDataBuffer(fits_data_buf_);
    }
    std::cout << "Exit constructor" << std::endl;
}

void fitsLoader::ldSeq() {
    for (auto &fits: listFits_) {
        fits.extractDataDirect(fits.naxis1());
        fits.reduceData();
    }
}

/*void fitsLoader::assembleAllTimes() {

    // Allocate memory for the buffer associated with MPI window
    size_t assembledDataLen = (nchans_ / world_size_) * 
                (numGlobFits_ * listFits_[0].dimTime());
    std::cout << "dimTime =  " << listFits_[0].dimTime();
    assembledDataBuffer = std::make_unique<float[]>(assembledDataLen);
    float* assembledDataBufferPtr = assembledDataBuffer.get();

    std::cout << "local size = " << assembledDataLen  << std::endl;
 
    MPI_Win win;
    MPI_Win_create(assembledDataBufferPtr, assembledDataLen * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);

    size_t contiguousChunkLen = (nchans_ / world_size_) * listFits_[0].dimTime();
    std::unique_ptr<float[]> contiguousData = std::make_unique<float[]>(contiguousChunkLen);
    float* contiguousDataPtr = contiguousData.get();

    for (int i = 0 ; i < listFits_.size() ; ++i) {

        // Global index of the ith fits file in listFits_
        int globalIndex = world_rank_ * numLocFits_ + i;
        std::cout << "Proc " << world_rank_ << " processing global fits file no. " << globalIndex << std::endl;

        auto& fits = listFits_[i];
        fits.extractDataDirect(fits.naxis1());
        fits.reduceData();

        std::cout << "Proc " << world_rank_ << " extracted & reduced global fits file no. " << globalIndex << std::endl;

        // Collecting data for a target rank. This does a partitioning 
        // of channels. All the time points in a fits file are dealt with here
        for (int target_rank = 0 ; target_rank < world_size_; ++target_rank) {

            int start_chan = target_rank * nchans_ / world_size_;
            int end_chan = start_chan + nchans_ / world_size_;
            
            std::cout << "Proc " << world_rank_ << " packing channels " << start_chan << "-" << end_chan << ", for local file " << i << std::endl;
            // Get the data from the channel chunk into the contiguous buffer
            //packChannelChunk(i, contiguousDataPtr, start_chan, end_chan);
            
            // offset required for the target rank based on the source
            size_t target_offset = globalIndex * contiguousChunkLen;

            MPI_Put(contiguousDataPtr, contiguousChunkLen, MPI_FLOAT, target_rank, target_offset, contiguousChunkLen, MPI_FLOAT, win);
            MPI_Win_fence(0, win);
        }

        
    } // end loop over fits

    MPI_Win_free(&win);
}   */

void fitsLoader::assembleAllTimesTest() {
    std::cout << "test function" << std::endl;

    // length of assembled data: channel_chunk_size * dimTimeGlobal
    size_t channelChunkSize = nchans_ / world_size_;
    size_t dimTimeTotal = numGlobFits_ * listFits_[0].dimTime();
    size_t assembledDataLen = channelChunkSize * dimTimeTotal;

    std::cout << "dimTime =  " << listFits_[0].dimTime() << std::endl;

    //assembledDataBuffer = std::make_unique<float[]>(assembledDataLen);
    assembledDataBuffer_ = static_cast<float*> (std::malloc(assembledDataLen * sizeof(float)));
    float* assembledDataBufferPtr = assembledDataBuffer_;//assembledDataBuffer.get();

    std::cout << "local size = " << assembledDataLen  << std::endl;
 
    //MPI_Win win;
    //MPI_Win_create(assembledDataBufferPtr, assembledDataLen * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    //MPI_Win_fence(0, win);

    size_t contiguousChunkLen = channelChunkSize * listFits_[0].dimTime();
    std::unique_ptr<float[]> contiguousData = std::make_unique<float[]>(contiguousChunkLen);
    float* contiguousDataPtr = contiguousData.get();

    for (int i = 0 ; i < listFits_.size() ; ++i) {

        // Global index of the ith fits file in listFits_
        int globalIndex = world_rank_ * numLocFits_ + i;
        std::cout << "Proc " << world_rank_ << " processing global fits file no. " << globalIndex << std::endl;

        auto& fits = listFits_[i];
        fits.extractDataDirect(fits.naxis1());
        fits.reduceData();

        std::cout << "Proc " << world_rank_ << " extracted & reduced global fits file no. " << globalIndex << std::endl;

        // Collecting data for a target rank. This does a partitioning 
        // of channels. All the time points in a fits file are dealt with here
        for (int target_rank = 0 ; target_rank < world_size_; ++target_rank) {

            int start_chan = target_rank * channelChunkSize;
            int end_chan = start_chan + channelChunkSize;
            
            std::cout << "Proc " << world_rank_ << " packing channels " << start_chan << "-" << end_chan << ", for local file " << i << std::endl;
            // Get the data from the channel chunk into the contiguous buffer
            packChannelChunk(i, contiguousDataPtr, start_chan, end_chan);

            std::cout << "Proc " << world_rank_ << " packing channels finished" << start_chan << "-" << end_chan << ", for local file " << i << std::endl;
            
            // offset required for the target rank based on the source
            size_t target_offset = globalIndex * contiguousChunkLen;

            //MPI_Put(contiguousDataPtr, contiguousChunkLen, MPI_FLOAT, target_rank, target_offset, contiguousChunkLen, MPI_FLOAT, win);
            //MPI_Win_fence(0, win);
        }

        
    } // end loop over fits

    //MPI_Win_free(&win);
}   

void fitsLoader::packChannelChunk(int i, float* contiguousDataPtr, int start_chan, int end_chan) {
    auto& fits = listFits_[i];
    float* data = fits.data();

    size_t channelChunkSize = end_chan - start_chan;

    for (size_t j = 0 ; j < fits.dimTime() ; ++j) {
        for (int chan = start_chan ; chan < end_chan ; ++chan) {
            contiguousDataPtr[j * (nchans_ / world_size_) + chan - start_chan] = data[j * nchans_ + chan];
        }
    }
}


