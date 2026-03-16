#include "barycenter.hpp"
#include "barycenter_presto_utils.h"
#include <cstdlib>
#include <cmath>
#include <vector>
#include <ranges>
#include <algorithm>
#include <tuple>
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

std::tuple<std::vector<int>, std::vector<double>, double> calcDelaysAndVels(char rastring[50], char decstring[50],
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

  // Save absolute barycentric MJD of first sample before converting to relative bins
  double blotoa = btoa[0];

  // Computing the differences in the topo and barycentric times.
  // The differences are converted to "bins" units.
  // The differences are shifted such that the first difference at
  // 0 is equal to zero. This is to align the  topo and bary
  // timeseries at the left end
  double dtmp = (btoa[0] - ttoa[0]);
  for (int ii = 0; ii < numbarypts; ii++)
    btoa[ii] = ((btoa[ii] - ttoa[ii]) - dtmp) * SECPERDAY / dt;

  ////////////////////////////////////////////////////////////////////////////////
  // btoa now contains entries in units of number of bins. They can be fractional!
  
  // Assuming original series length = N
  
  // If the entries in btoa were calculated for all indices, and not
  // just numbarypts indices, the entries in btoa would be
  //               d_0, d_1, d_2, ...., d_{N-1}
  // These entries represent the following map from topocentric bin indices
  // to barycentric bin indices
  //               i -> i + d_i
  // that is, if the original topocentric series is
  //              T_0, T_1, T_2, ......, T_{N-1}
  // The same series in barycentric time is
  //              B_0, B_{1+d_1}, B_{2+d_2}, ...., B_{N-1 + d_{N-1}}
  // or 
  //              T_i = B_{i + d_i}

  // The differences d_i are fractional at this point and can be positive or 
  // negative. The goal of barycentering is to have the series at regularly 
  // spaced intervals in the barycentric time, that is going from 
  //              B_0, B_{1+d_1}, B_{2+d_2}, ...., B_{N-1 + d_{N-1}}
  // to
  //              B_0, B_1, B_2, ....
  
  // What is actually done in this code is not computing the d values for all 
  // points, but only for a small number of points "numbarypts". So the d values
  // correspond to the shift in terms of bins at various points in the original 
  // time series
  //              T_0, T_1, T_2, ......, T_{N-1}
  //              d_0        d_1        d_2 ....

  // The code also makes another assumption about d values which is implicit:
  // It assumes that d values are non increasing/decreasing

  // Since d essentially tells the change in the timeseries size until that point
  // we calculate the position where to make those changes. For example, let's say 
  // d1 = 5. This means by that point, the barycentered timeseries is 5 bins longer
  // . These 5 bins have to be added somewhere and the code below calculates that.
  // Similarly for negative d values, the timeseries shrinks and the procedure below 
  // calculates places where bins are to be eliminated. The changes are made at 
  // topocentric bin positions where d values would be multiples of 0.5! 

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

  return std::make_tuple(diffbins, voverc, blotoa);
} 
