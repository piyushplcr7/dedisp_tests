#include "barycenter.hpp"
#include "barycenter_presto_utils.h"
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ranges>
#include <algorithm>
#include <iostream>

void createResampleMap(int N_in, int N_out, int* diffbinptr, int numdiffbins, int* inForOut, int* insertPositions) {
  int offset = 0;
  for (int i = 0 ; i < N_in ; ++i) {
    // Check for possible insertion or deletion
    if (std::abs(*diffbinptr) == i) {
      // insertion
      if (*diffbinptr > 0) {
        // duplication 
        inForOut[i + offset] = i;
        *(insertPositions++) = i+offset;
        offset++;
        diffbinptr++;
      }
      // deletion
      else {
        offset--;
        diffbinptr++;
        continue;
      }
    }
    inForOut[i + offset] = i;
  }
}

std::pair<std::vector<int>, std::vector<double>> calcDelaysAndVels(char rastring[50], char decstring[50], 
                                                                   char obs[3], char ephem[10], int N, 
                                                                   double dt, double tlotoa){
  // Number of points where topo and bary times are computed using tempo
  int numbarypts = (int) (dt * N * 1.1 / TDT + 5.5) + 1;

  // Creating vectors for recording output of barycenter function from presto
  std::vector<double> ttoa(numbarypts);
  std::vector<double> btoa(numbarypts);
  std::vector<double> voverc(numbarypts);

  // Generating points where topo and bary times are computed
  for (int i = 0 ; i < numbarypts ; ++i) 
    ttoa[i] = tlotoa + TDT * i / SECPERDAY;

  // Calling the barycenter function from presto to get the topo/bary times
  barycenter(ttoa.data(), btoa.data(), voverc.data(), numbarypts, rastring, decstring, obs, ephem);  

  // Computing the differences in the topo and barycentric times.
  // The differences are converted to "bins" units.
  // The differences are shifted such that the first differece at 
  // 0 is equal to zero. This is to align the timeseries at the 
  // left end
  double dtmp = (btoa[0] - ttoa[0]);
  for (int ii = 0; ii < numbarypts; ii++)
    btoa[ii] = ((btoa[ii] - ttoa[ii]) - dtmp) * SECPERDAY / dt;

  // Size difference between the two timeseries in terms of bins 
  int numdiffbins = labs(lrint(btoa[numbarypts - 1])) + 1;
  std::vector<int> diffbins(numdiffbins);

  int oldbin = 0, currentbin;
  double lobin, hibin, calcpt;

  int *diffbinptr = diffbins.data();

  for (int ii = 1; ii < numbarypts; ii++) {
    currentbin = lrint(btoa[ii]);
    if (currentbin != oldbin) {
      if (currentbin > 0) {
        calcpt = oldbin + 0.5;
        lobin = (ii - 1) * TDT / dt;
        hibin = ii * TDT / dt;
      } else {
        calcpt = oldbin - 0.5;
        lobin = -((ii - 1) * TDT / dt);
        hibin = -(ii * TDT / dt);
      }
      while (fabs(calcpt) < fabs(btoa[ii])) {
        /* Negative bin number means remove that bin */
        /* Positive bin number means add a bin there */
        *diffbinptr = lrint(LININTERP(calcpt, btoa[ii - 1],
                                                btoa[ii], lobin,
                                                hibin));
        diffbinptr++;
        calcpt = (currentbin > 0) ? calcpt + 1.0 : calcpt - 1.0;
      }
      oldbin = currentbin;
    }
  }

  // Process additions so that no index beyond max length are there
  auto it = std::find_if(diffbins.begin(), diffbins.end(), [&](int x) {return abs(x) >= N;});

  if (it == diffbins.end()) {
    std::cout << "Warning: no elements removed from diffbins. Not expected!" << std::endl;
  }

  // Erasing unneeded indices
  diffbins.erase(it, diffbins.end());

  return std::make_pair(diffbins, voverc);
} 
