#include <iostream>
#include "fits/fits.hpp"
#include <memory>
#include <new>
#include <mpi.h>
#include <vector>
#include <string>
#include "fitscontainer.hpp"
#include "matrix_view.hpp"

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int numfiles = argc - 1;
/*     std::cout << "Numfiles = " << numfiles << std::endl;

    std::vector<std::string> listFitsNames(numfiles);

    for (int i = 0 ; i < numfiles ; ++i) {
        std::cout << argv[i+1] << std::endl;
        listFitsNames[i] = std::string(argv[i+1]);
    }

    dataLoader container(listFitsNames, world_rank, world_size, 1);

    std::cout << "assembling distributed data using MPI on " << world_rank << std::endl;
    container.assembleAllTimes();
    std::cout << "assembled distributed data using MPI on " << world_rank << std::endl;

    // Now generate one fits object
    Fits standaloneFits(listFitsNames[0].c_str());

    // Buffer to store its data
    std::unique_ptr<float[]> data = std::make_unique<float[]>(standaloneFits.getNumElements());

    std::cout << "Getting data in the standalone fits object on " << world_rank << std::endl;
    {
        std::unique_ptr<unsigned char, void (*)(unsigned char*)> alignedBuf(
            static_cast<unsigned char*>( ::operator new(standaloneFits.fileSizeAligned() , std::align_val_t(4096))),
            [] (unsigned char* x) { ::operator delete(x, std::align_val_t(4096)); }
        );

        standaloneFits.setAlignedFileSizeBuffer(alignedBuf.get());
        standaloneFits.setDataBuffer(data.get());

        // Extract and reduce the data
        standaloneFits.extractDataDirect();
        standaloneFits.reduceData();
    }

    std::cout << "standalone fits created on " << world_rank << std::endl;

    // Get data view on the assembled data and standalone fits
    matrixView<float> assembled = container.getAssembledData();
    matrixView<float> standalone = standaloneFits.dataView();

    std::cout << "assembled.rows() = " << assembled.rows() << ", assembled.cols() = " << assembled.cols() << std::endl;
    std::cout << "standalone.rows() = " << standalone.rows() << ", standalone.cols() = " << standalone.cols() << std::endl;
    //exit(-1);
    size_t start_chan = container.startChan();
    // Comparing the assembled with the standalone
    for (size_t i = 0 ; i < assembled.rows() ; ++i) {
        for (size_t j = 0 ; j < assembled.cols() ; ++j) {
            if ( std::abs(assembled(i,j) - standalone(i % standalone.rows() , j + start_chan) ) > 1e-5 ) {
                std::cout << "mismatch at " << i << ", " << j << "on rank " << world_rank << std::endl;
            }
        }
    }

    std::cout << "Test passed! on " << world_rank << std::endl;

    std::cout << "Just before MPI_Finalize() on proc " << world_rank << std::endl; */
    MPI_Finalize();
    return 0; 
}