#ifndef BARYCENTERHPP
#define BARYCENTERHPP

#include <cstring>
#include <vector>
#include <chrono>
#include <iostream>

// Resample map
void createResampleMap(int N_in, int N_out, int* diffbinptr, int numdiffbins, int* inForOut, int* insertPositions);

// Calculate delays and velocities
std::pair<std::vector<int>, std::vector<double>> calcDelaysAndVels(char rastring[50], char decstring[50], char obs[3], char ephem[10], int N, double dt, double tlotoa);

// Barycenter assuming that channels is the fastest changing index
// this function is doing the simple duplication strategy
template <typename T>
void resampleTimeSeriesDuplication(std::vector<T>& data_in, std::vector<T>& data_out, 
                          size_t nchans, int Nout, std::vector<int>& inForOut) {
  #pragma omp parallel for schedule(static)
  for (int i = 0 ; i < Nout ; ++i) {
    T* out = data_out.data() + i * nchans;
    T* in = data_in.data() + inForOut[i] * nchans;
    std::memcpy(out, in, nchans * sizeof(T));
  }
}

// Barycenter assuming that channels is the fastest changing index
// this function is doing the simple duplication strategy
template <typename T>
void resampleTimeSeriesMovAvg(std::vector<T>& data_in, std::vector<T>& data_out, 
                                size_t nchans, size_t Nout, std::vector<int>& inForOut, 
                                int numAdditions, std::vector<int>& insertPositions) {

  // Copy the data using the duplication strategy first
  resampleTimeSeriesDuplication(data_in, data_out, nchans, Nout, inForOut);
 
  // Compute moving averages where required
  std::vector<float> movavg(nchans,0.);

  int t = 0;
  for (int i = 0 ; i < numAdditions ; ++i) {
    for (; t < insertPositions[i] ; ++t) {
      for (int c = 0 ; c < nchans ; ++c) {
        movavg[c] += data_in[t * nchans + c];
      }
    }
    // Moving avg accumulated, write to data
    for (int c = 0 ; c < nchans ; ++c) {
      data_out[insertPositions[i] * nchans + c] = movavg[c]/t;
    }
  }
}

// Barycenter assuming that channels is the fastest changing index
// this function is doing the simple duplication strategy
template <typename T>
void resampleTimeSeriesLocAvg(std::vector<T>& data_in, std::vector<T>& data_out, 
                                size_t nchans, int Nout, std::vector<int>& inForOut, 
                                std::vector<int>& insertPositions, int window) {

  // Copy the data using the duplication strategy first
  resampleTimeSeriesDuplication(data_in, data_out, nchans, Nout, inForOut);

  int numAdditions = insertPositions.size();
  
  #pragma omp parallel for
  for (int i = 0 ; i < numAdditions ; ++i) {
    std::vector<T> locavg(nchans,0.);
    int insertPosition = insertPositions[i];
    int lowbin = std::max(0, insertPosition-window);
    int locsize = insertPosition-lowbin;
    
    // Calculate the local average
    for (int j = lowbin ; j < insertPosition ; ++j) {
      #pragma GCC ivdep
      for (int c = 0 ; c < nchans ; ++c) {
        locavg[c] += data_in[j * nchans + c];
      }
    }

    // Assign the local average at insertPosition
    #pragma omp simd
    for (int c = 0 ; c < nchans ; ++c) {
      data_out[insertPosition * nchans + c] = locavg[c]/locsize;
    }
  }

}

#endif
