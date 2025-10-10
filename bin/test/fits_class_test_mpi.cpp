#include <iostream>
#include "fits/fits.hpp"
#include <memory>
#include <new>
#include <mpi.h>

struct AlignedDeleter {
    std::size_t align;
    void operator()(unsigned char* p) const noexcept {
        ::operator delete(p, std::align_val_t{align}); // matches allocation
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const char* filename = "/scratch/panchal/G0057_1255444104_00:30:27.42_+04:51:39.73_ch109-132_0001.fits";
    Fits obj(filename);

    //if (world_rank == 0)
        obj.setVerbosity(1);
    obj.readDataMPI(world_rank, world_size,0);

    unsigned char* mpi_rawdata_buffer = obj.mpiRawdataBuffer();

    std::cout << "Inside process: " << world_rank << ", local channel start = " << obj.chanStartLocal() << ", num channels = " << obj.nchanLocal() << std::endl;
    for (int i =  50000 ; i < 50010 ; ++i) {
        std::cout << "local buffer[" << i << "] = " << (int)mpi_rawdata_buffer[i] << std::endl;
    }

    // Allocate buffer for reduced data
    size_t reduced_data_size = (size_t)obj.nchanLocal() * obj.nsblk() * obj.naxis2() * sizeof(float);
    float* mpi_reduced_data_buffer = (float*)malloc(reduced_data_size);
    obj.reduceDataMPI(world_rank, world_size, mpi_reduced_data_buffer);

    free(mpi_reduced_data_buffer);

    MPI_Finalize();


    return 0; 
}