#include <iostream>
#include "fits/fits.hpp"
#include <memory>
#include <new>
#include <mpi.h>
#include <vector>
#include <string>
#include "fits/fitscontainer.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int numfiles = argc - 1;
    std::cout << "Numfiles = " << numfiles << std::endl;

    std::vector<std::string> listFitsNames(numfiles);

    for (int i = 0 ; i < numfiles ; ++i) {
        std::cout << argv[i+1] << std::endl;
        listFitsNames[i] = std::string(argv[i+1]);
    }

    float *a, *b;
    unsigned char *c;
    {
        fitsLoader container(listFitsNames, world_rank, world_size);
        container.assembleAllTimesTest();
        a = container.fits_data_buf_;
        b = container.assembledDataBuffer_;
        c = container.aligned_buf_;
    }

    std::free(a);
    std::cout << "freed a on " << world_rank << std::endl;
    std::free(b);
    std::cout << "freed b on " << world_rank << std::endl;
    std::free(c);
    std::cout << "freed c on " << world_rank << std::endl;

    std::cout << "Just before MPI_Finalize() on proc " << world_rank << std::endl;
    MPI_Finalize();
    return 0; 
}