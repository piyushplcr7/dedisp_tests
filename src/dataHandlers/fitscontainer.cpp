#include "fitscontainer.hpp"
#include "fits/fits.hpp"
#include "fil/fil.hpp"
#include "fits/fits.hpp"
#include <vector>
#include <iostream>
#include <memory>
#include <mpi.h>
#include "matrix_view.hpp"
#include "hwloc_utils.hpp"
#include <chrono>
//#include "barycenter_presto_utils.h"
#include "barycenter.hpp"
#include <ranges>
#include <algorithm>
#include <iomanip>

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

dataLoader::dataLoader(std::vector<std::string>& listFitsNames, int world_rank, int world_size, int downsamp, int nbits)
        :world_rank_(world_rank), world_size_(world_size), downsamp_(downsamp), nbits_(nbits) { 
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
    listFiles_.reserve(numLocFits_);

    const std::string& firstFile = listFitsNames[0];
    if (firstFile.size() >= 4 && firstFile.substr(firstFile.size() - 4) == ".fil") {
        std::cout << "Input files are filterbank files" << std::endl;
        firstFile_ = std::make_unique<Fil>(firstFile.c_str());
        if (nbits_ == 0) {
            std::cout << "nbits not specified, defaulting to " << nbits_ << " bits for filterbank files" << std::endl;
        }
        nbits_ = nbits_ == 0 ? 8 : nbits_; // default to 8 bits for filterbank files if nbits not specified 
    } else if (firstFile.size() >= 5 && firstFile.substr(firstFile.size() - 5) == ".fits") {
        std::cout << "Input files are fits files" << std::endl;
        firstFile_ = std::make_unique<Fits>(firstFile.c_str());
        if (nbits_ == 0) {
            std::cout << "nbits not specified, defaulting to " << nbits_ << " bits for fits files" << std::endl;
        }
        nbits_ = nbits_ == 0 ? 32 : nbits_; // default to 32 bits for fits files if nbits not specified 
    }

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        const std::string& fname = listFitsNames[world_rank_ * numLocFits_ + i];
        if (fname.size() >= 4 && fname.substr(fname.size() - 4) == ".fil") {
            listFiles_.emplace_back(std::make_unique<Fil>(fname.c_str()));
        } else {
            listFiles_.emplace_back(std::make_unique<Fits>(fname.c_str()));
        }
    }
    nsampsGlobal_ = firstFile_->dimTime(downsamp_) * numGlobFits_;

    if (!listFiles_[0]->checkDownsamp(downsamp_)) {
        std::cerr << "Downsampling factor "
        << downsamp_ << " not allowed! Please choose a different value" << std::endl;
        exit(-1);
    }

#ifdef TESTDEDISP_DEBUG
    if (world_rank_ == 0) {
        firstFile_->printInfo();
    }
#endif

    /* nchans_ = listFiles_[0].nchan();

    if (nchans_ % world_size_ != 0) {
        std::cerr << "Error, nchans not divisible by npes = " << world_size_ << std::endl;
        exit(-1);
    }

    channelChunkSize_ = nchans_ / world_size_;

    // length of assembled data: channel_chunk_size * dimTimeGlobal
    contiguousChunkLen_ = channelChunkSize_ * listFiles_[0].dimTime(downsamp_);
    assembledDataLen_ = contiguousChunkLen_ * numGlobFits_;

    start_chan_ = world_rank_ * channelChunkSize_;
    end_chan_ = start_chan_ + channelChunkSize_ ; */

}

void dataLoader::allocContiguousAssemblyBuf() {
    if (assembledDataLen_ == 0) {
        std::cerr << "assembled data size = 0! Container usage has not been determined!" << std::endl;
        exit(-1);
    }

    assembledDataBuffer_ = std::make_unique<float[]>(assembledDataLen_ / (32 / nbits_));

}

void dataLoader::ldSeq(size_t chunksize, int poln) {
#ifdef TESTDEDISP_DEBUG
    std::cout << "ldSeq() called. Container will contain only the data from the local fits files. No reshuffling! " << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
#endif

    nchans_ = listFiles_[0]->nchan();
    nsampsLocal_ = listFiles_[0]->dimTime(downsamp_) * numLocFits_;
    assembledDataLen_ = nchans_ * nsampsLocal_;

    allocContiguousAssemblyBuf();

    // All channels owned
    start_chan_ = 0;
    end_chan_ = nchans_;

    if (assembledDataBuffer_.get() == nullptr) {
        std::cerr << "Assembled data buffer not allocated!" << std::endl;
        exit(-1);
    }

    std::unique_ptr<unsigned char, void (*)(unsigned char*)> alignedBuf(
        static_cast<unsigned char*>( ::operator new(listFiles_[0]->fileSizeAligned() , std::align_val_t(4096))),
        [] (unsigned char* x) { ::operator delete(x, std::align_val_t(4096)); }
    );

#ifdef TESTDEDISP_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "ldSeq(): Allocated aligned buffer in " << diff.count() << " seconds" << std::endl;
#endif

    for (int i = 0; i < (int)listFiles_.size(); ++i) {
        int bytes_per_samp = nbits_ / 8;
        size_t offset = i * listFiles_[0]->getNumElements(downsamp_) * bytes_per_samp;
        auto& f = listFiles_[i];

        auto start1 = std::chrono::high_resolution_clock::now();
        f->extractDataDirect(alignedBuf.get(), chunksize);
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff1 = end1 - start1;
        std::cout << "Read file " << f->getFilename() << ", in " << diff1.count() << " sec, speed: " << (double)f->fileSize()/(1<<20)/diff1.count() << " MBps" << std::endl;
    
        auto* dataPtr = reinterpret_cast<unsigned char*>(assembledDataBuffer_.get()) + offset;
        f->reduceData(dataPtr, nbits_, poln, downsamp_);
    }
}

void dataLoader::barycenter() {
    // Barycentric corrections must be computed against the FULL global time
    // series (start epoch + total length), not the local chunk. firstFile_
    // is opened on every rank in the constructor so this is deterministic
    // and identical across ranks without any communication.
    std::vector<double> voverc;
    std::tie(diffbins_, voverc, blotoa_) = calcDelaysAndVels(firstFile_->rightAscension(),
                                                    firstFile_->declination(),
                                                    firstFile_->obs(),
                                                    firstFile_->ephem(),
                                                    nsampsGlobal_,
                                                    firstFile_->sampletime(downsamp_),
                                                    firstFile_->epoch());

    // Calculate the net shift
    int net_shift = 0;
    for (auto x: diffbins_) {
        if (x > 0 && abs(x) < nsampsGlobal_) {
            net_shift++;
        }
        else if (x < 0 && abs(x) < nsampsGlobal_) {
            net_shift--;
        }
    }

    // calculate max, min and avg voverc
    for (int i = 0 ; i < voverc.size(); i++) {
        if (voverc[i] > maxvoverc_)
            maxvoverc_ = voverc[i];
        if (voverc[i] < minvoverc_)
            minvoverc_ = voverc[i];

        avgvoverc_ += voverc[i];
    }
    avgvoverc_ /= voverc.size();

    std::cout << "Average topocentric velocity (c) = " << std::setprecision(7) << avgvoverc_ << std::endl;
    std::cout << "Maximum topocentric velocity (c) = " << std::setprecision(7) << maxvoverc_ << std::endl;
    std::cout << "Minimum topocentric velocity (c) = " << std::setprecision(7) << minvoverc_ << std::endl;

    size_t N_out = nsampsGlobal_ + net_shift;
    // Count bin removals and additions
    int removals = std::ranges::count_if(diffbins_, [](int x){ return x < 0; });
    int additions = std::ranges::count_if(diffbins_, [](int x){ return x > 0; });
    std::cout << "removals: " << removals << ", additions: " << diffbins_.size()-removals << std::endl;

    // Creat resample map inForOut
    inForOut_.resize(N_out);
    insertPositions_.resize(additions);
    createResampleMap(nsampsGlobal_, N_out, diffbins_.data(), diffbins_.size(), inForOut_.data(), insertPositions_.data());

    //barycentered_data = new float[N_out * nchansLocal()];

    //resampleTimeSeriesLocAvg(assembledDataBuffer_.get(), barycentered_data, nchansLocal(), N_out, 
    //                           inForOut, insertPositions, 10000);

    // Deallocate the assembled data buffer and set it to the barycentered data buffer
    /* assembledDataBuffer_.reset(barycentered_data);
    nsampsLocal_ = N_out; */
}

void dataLoader::parLoad() {

}

void dataLoader::reduceInAssemblyBuf() {
    if (assembledDataBuffer_.get() == nullptr) {
        std::cerr << "Assembled data buffer not allocated!" << std::endl;
        exit(-1);
    }
    if (world_size_ > 1) {
        std::cerr << "Using reduceInAssemblyBuf is not meant for parallel runs!" << std::endl;
        exit(-1);
    }

    // With the new buffer-passing API, callers pass the output buffer
    // directly to reduceData() — no per-file setDataBuffer() call needed here.
}

void dataLoader::assembleAllTimes() {
#ifdef TESTDEDISP_DEBUG
    std::cout << "Shuffling data between MPI procs to get all times but channel subsets" << std::endl;
#endif
    nchans_ = listFiles_[0]->nchan();

    if (nchans_ % world_size_ != 0) {
        std::cerr << "Error, nchans not divisible by npes = " << world_size_ << std::endl;
        exit(-1);
    }

    // The container gets a chunk of channels
    nchans_ = nchans_ / world_size_;

    // length of assembled data: channel_chunk_size * dimTimeGlobal
    contiguousChunkLen_ = nchans_ * listFiles_[0]->dimTime(downsamp_);
    assembledDataLen_ = contiguousChunkLen_ * numGlobFits_;

    start_chan_ = world_rank_ * nchans_;
    end_chan_ = start_chan_ + nchans_;

    allocContiguousAssemblyBuf();

    std::unique_ptr<unsigned char, void (*)(unsigned char*)> aligned_buf(
        static_cast<unsigned char*>(::operator new(listFiles_[0]->fileSizeAligned(), std::align_val_t(ALIGNMENT))),
        [](unsigned char* x){ ::operator delete(x, std::align_val_t(ALIGNMENT)); }
    );

    // Allocate space for the reduced data (reused for each file)
    std::unique_ptr<float[]> data_buf = std::make_unique<float[]>(listFiles_[0]->getNumElements());

    const matrixView<float> dataView(data_buf.get(), listFiles_[0]->dimTime(), listFiles_[0]->nchan());

    float* assembledDataBufferPtr = assembledDataBuffer_.get();

    MPI_Win win;
    MPI_Win_create(assembledDataBufferPtr, assembledDataLen_ * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);

    std::unique_ptr<float[]> contiguousData = std::make_unique<float[]>(contiguousChunkLen_);
    matrixView<float> contiguousDataView(contiguousData.get(), listFiles_[0]->dimTime(), nchans_);

    for (unsigned int i = 0 ; i < listFiles_.size() ; ++i) {

        int globalFitsIndex = world_rank_ * numLocFits_ + i;

        auto& f = listFiles_[i];
        f->extractDataDirect(aligned_buf.get(), HALF_MAX_CHUNKSIZE);
        f->reduceData((unsigned char*)data_buf.get(), nbits_, 0, downsamp_);

        // Collecting data for a target rank. This does a partitioning 
        // of channels. All the time points in a fits file are dealt with here
        for (int target_rank = 0 ; target_rank < world_size_; ++target_rank) {

            int start_chan = target_rank * nchans_;
            int end_chan = start_chan + nchans_;
            
            // Get the data from the channel chunk into the contiguous buffer
            packChannelChunk(dataView, contiguousDataView, start_chan, end_chan);
            
            // offset required for the target rank based on the source
            size_t target_offset = globalFitsIndex * contiguousChunkLen_;

            MPI_Put_split_Float(contiguousDataView.data(), contiguousChunkLen_, target_rank, target_offset, win);
            MPI_Win_fence(0, win);
        }
    } // end loop over fits

    MPI_Win_free(&win);
}   

void dataLoader::assembleAllTimesAsync() {
    /* allocContiguousAssemblyBuf();

    // Allocate aligned buffer once which will be reused for all Fits extraction
    std::unique_ptr<unsigned char, void (*)(unsigned char*)> aligned_buf(
        static_cast<unsigned char*>(::operator new(listFiles_[0].fileSizeAligned(), std::align_val_t(ALIGNMENT))),
        [](unsigned char* x){ ::operator delete(x, std::align_val_t(ALIGNMENT)); }
    );

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFiles_[i].setAlignedFileSizeBuffer(aligned_buf.get());
    }

    // Allocate space for the reduced data
    std::unique_ptr<float[]> fits_data_buf = std::make_unique<float[]>(listFiles_[0].getNumElements());

    for (int i = 0 ; i < numLocFits_ ; ++i) {
        listFiles_[i].setDataBuffer(fits_data_buf.get());
    }

    float* assembledDataBufferPtr = assembledDataBuffer_.get();
 
    MPI_Win win;
    MPI_Win_create(assembledDataBufferPtr, assembledDataLen_ * sizeof(float), sizeof(float), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    
    std::unique_ptr<float[]> contiguousData = std::make_unique<float[]>(contiguousChunkLen_);
    matrixView<float> contiguousDataView(contiguousData.get(), listFiles_[0].dimTime(), nchans_);

    for (unsigned int i = 0 ; i < listFiles_.size() ; ++i) {

        // Global index of the ith fits file in listFiles_
        int globalFitsIndex = world_rank_ * numLocFits_ + i;

        auto& fits = listFiles_[i];
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

    MPI_Win_free(&win); */
}   

bool dataLoader::segmentsEqualLength() const {
    if (listFiles_.empty()) return true;
    size_t ref = listFiles_[0]->dimTime(downsamp_);
    for (const auto& f : listFiles_) {
        if (f->dimTime(downsamp_) != ref) return false;
    }
    return true;
}

unsigned char* dataLoader::getSegmentPtr(int i) const {
    if (assembledDataBuffer_.get() == nullptr) {
        std::cerr << "Assembled data buffer not allocated!" << std::endl;
        exit(-1);
    }
    if (i < 0 || i >= numLocFits_) {
        std::cerr << "Segment index " << i << " out of range [0, " << numLocFits_ << ")" << std::endl;
        exit(-1);
    }
    int bytes_per_samp = nbits_ / 8;
    size_t offset = static_cast<size_t>(i) * listFiles_[0]->getNumElements(downsamp_) * bytes_per_samp;
    return reinterpret_cast<unsigned char*>(assembledDataBuffer_.get()) + offset;
}

void dataLoader::packChannelChunk(matrixView<float> data, matrixView<float> contiguousDataView, int start_chan, int end_chan) {
    for (size_t j = 0 ; j < data.rows() ; ++j) {
        for (int chan = start_chan ; chan < end_chan ; ++chan) {
            contiguousDataView(j, chan - start_chan) = data(j, chan);
        }
    }
}




