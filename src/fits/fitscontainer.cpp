#include "fitscontainer.hpp"
#include <vector>
#include <iostream>
#include <memory>
#include <mpi.h>
#include "matrix_view.hpp"

/*
 * Safe large MPI_Put
 *
 * - Works for arbitrarily large buffers
 * - Respects MPI's int-count limitation
 * - Assumes:
 *     - window already created
 *     - access epoch already open (fence or lock)
 *     - disp_unit corresponds to element size
 */
inline void MPI_Put_split_Float(
    const float* origin_buf,
    size_t        origin_count,
    int           target_rank,
    size_t        target_disp,
    MPI_Win       win
) {
    const size_t MAX_ELEMS = static_cast<size_t>(INT_MAX);

    size_t offset = 0;
    size_t remaining = origin_count;

    while (remaining > 0) {
        size_t chunk = std::min(remaining, MAX_ELEMS);

        MPI_Put(origin_buf + offset,
                static_cast<int>(chunk),
                MPI_FLOAT,
                target_rank,
                static_cast<MPI_Aint>(target_disp + offset),
                static_cast<int>(chunk),
                MPI_FLOAT,
                win);

        offset    += chunk;
        remaining -= chunk;
    }
}

fitsLoader::fitsLoader(std::vector<std::string>& listFitsNames, int world_rank, int world_size) { 
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
    listFits_.reserve(numLocFits_);

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFits_.emplace_back(listFitsNames[world_rank_ * numLocFits_ + i].c_str());
    }

    nchans_ = listFits_[0].nchan();

    if (nchans_ % world_size_ != 0) {
        std::cerr << "Error, nchans not divisible by npes = " << world_size_ << std::endl;
        exit(-1);
    }

    channelChunkSize_ = nchans_ / world_size_;

    // length of assembled data: channel_chunk_size * dimTimeGlobal
    contiguousChunkLen_ = channelChunkSize_ * listFits_[0].dimTime();
    assembledDataLen_ = contiguousChunkLen_ * numGlobFits_;

    assembledDataBuffer_ = std::make_unique<float[]>(assembledDataLen_);
    assembledData_ = matrixView<float> 
        (assembledDataBuffer_.get(),  
        listFits_[0].dimTime() * numGlobFits_,
        (size_t)channelChunkSize_
        );

    start_chan_ = world_rank_ * channelChunkSize_;
    end_chan_ = start_chan_ + channelChunkSize_ ;
}

void fitsLoader::ldSeq() {
    for (auto &fits: listFits_) {
        fits.extractDataDirect(fits.naxis1());
        fits.reduceData();
    }
}

void fitsLoader::assembleAllTimes() {
    // Allocate aligned buffer once which will be reused for all Fits extraction
    std::unique_ptr<unsigned char, void (*)(unsigned char*)> aligned_buf(
        static_cast<unsigned char*>(::operator new(listFits_[0].fileSizeAligned(), std::align_val_t(ALIGNMENT))),
        [](unsigned char* x){ ::operator delete(x, std::align_val_t(ALIGNMENT)); }
    );

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFits_[i].setAlignedFileSizeBuffer(aligned_buf.get());
    }

    // Allocate space for the reduced data
    std::unique_ptr<float[]> fits_data_buf = std::make_unique<float[]>(listFits_[0].getNumElements());

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFits_[i].setDataBuffer(fits_data_buf.get());
    }

    float* assembledDataBufferPtr = assembledDataBuffer_.get();
 
    MPI_Win win;
    MPI_Win_create(assembledDataBufferPtr, assembledDataLen_ * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    
    std::unique_ptr<float[]> contiguousData = std::make_unique<float[]>(contiguousChunkLen_);
    matrixView<float> contiguousDataView(contiguousData.get(), listFits_[0].dimTime(), channelChunkSize_);

    for (unsigned int i = 0 ; i < listFits_.size() ; ++i) {

        // Global index of the ith fits file in listFits_
        int globalFitsIndex = world_rank_ * numLocFits_ + i;

        auto& fits = listFits_[i];
        fits.extractDataDirect(fits.naxis1());
        fits.reduceData();

        // Collecting data for a target rank. This does a partitioning 
        // of channels. All the time points in a fits file are dealt with here
        for (int target_rank = 0 ; target_rank < world_size_; ++target_rank) {

            int start_chan = target_rank * channelChunkSize_;
            int end_chan = start_chan + channelChunkSize_;
            
            // Get the data from the channel chunk into the contiguous buffer
            packChannelChunk(i, contiguousDataView, start_chan, end_chan);
            
            // offset required for the target rank based on the source
            size_t target_offset = globalFitsIndex * contiguousChunkLen_;

            MPI_Put_split_Float(contiguousDataView.data(), contiguousChunkLen_, target_rank, target_offset, win);
            MPI_Win_fence(0, win);
        }
    } // end loop over fits

    MPI_Win_free(&win);
}   

void fitsLoader::assembleAllTimesAsync() {

    // Allocate aligned buffer once which will be reused for all Fits extraction
    std::unique_ptr<unsigned char, void (*)(unsigned char*)> aligned_buf(
        static_cast<unsigned char*>(::operator new(listFits_[0].fileSizeAligned(), std::align_val_t(ALIGNMENT))),
        [](unsigned char* x){ ::operator delete(x, std::align_val_t(ALIGNMENT)); }
    );

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFits_[i].setAlignedFileSizeBuffer(aligned_buf.get());
    }

    // Allocate space for the reduced data
    std::unique_ptr<float[]> fits_data_buf = std::make_unique<float[]>(listFits_[0].getNumElements());

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFits_[i].setDataBuffer(fits_data_buf.get());
    }

    float* assembledDataBufferPtr = assembledDataBuffer_.get();
 
    MPI_Win win;
    MPI_Win_create(assembledDataBufferPtr, assembledDataLen_ * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    
    std::unique_ptr<float[]> contiguousData = std::make_unique<float[]>(contiguousChunkLen_);
    matrixView<float> contiguousDataView(contiguousData.get(), listFits_[0].dimTime(), channelChunkSize_);

    for (unsigned int i = 0 ; i < listFits_.size() ; ++i) {

        // Global index of the ith fits file in listFits_
        int globalFitsIndex = world_rank_ * numLocFits_ + i;

        auto& fits = listFits_[i];
        fits.extractDataDirect(fits.naxis1());
        fits.reduceData();

        // Collecting data for a target rank. This does a partitioning 
        // of channels. All the time points in a fits file are dealt with here
        for (int target_rank = 0 ; target_rank < world_size_; ++target_rank) {

            int start_chan = target_rank * channelChunkSize_;
            int end_chan = start_chan + channelChunkSize_;
            
            // Get the data from the channel chunk into the contiguous buffer
            packChannelChunk(i, contiguousDataView, start_chan, end_chan);
            
            // offset required for the target rank based on the source
            size_t target_offset = globalFitsIndex * contiguousChunkLen_;

            MPI_Put_split_Float(contiguousDataView.data(), contiguousChunkLen_, target_rank, target_offset, win);
            MPI_Win_fence(0, win);
        }
    } // end loop over fits

    MPI_Win_free(&win);
}   

void fitsLoader::packChannelChunk(int i, matrixView<float> contiguousDataView, int start_chan, int end_chan) {
    auto& fits = listFits_[i];
    const matrixView<float> data(fits.data(), fits.dimTime(), fits.nchan());

    for (size_t j = 0 ; j < fits.dimTime() ; ++j) {
        for (int chan = start_chan ; chan < end_chan ; ++chan) {
            contiguousDataView(j, chan - start_chan) = data(j,chan);
        }
    }
}




