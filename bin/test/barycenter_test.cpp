#include "barycenter_presto_utils.h"
#include "barycenter.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <ranges>
#include <algorithm>
#include <fstream>
#include <iomanip>

int main() {
    // Parameters for the data
    int N = 2000000;
    size_t nchans = 3072;
    float fillval = 1.;
    // Parameters important for computing barycentric correction
    double dt = 0.0001;
    int numbarypts = (int) (dt * N * 1.1 / TDT + 5.5) + 1;
    char rastring[50] = "00:30:27.4200";
    char decstring[50] = "04:51:39.7300";
    char obs[3] = "MW";
    char ephem[10] = "DE405";
    double tlotoa = 58774.607894;

    // Generate synthetic data and time
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<float> data(N * nchans, fillval);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "Generation of synthetic data took " << diff.count() << " seconds\n";

    start = std::chrono::high_resolution_clock::now();
    std::vector<double> voverc;
    std::vector<int> diffbins;
    double blotoa;
    std::tie(diffbins, voverc, blotoa) = calcDelaysAndVels(rastring, decstring, obs, ephem, N, dt, tlotoa);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "calcDelaysAndVels took " << diff.count() << " seconds\n";

    // Making diffbins positive for testing
    for (auto& x: diffbins) {
        x = abs(x);
    }

    int numdiffbins = diffbins.size();
    // The last entry of btoa denotes the bin drift between the two timeseries at the end
    // , when the starting bins are aligned. 
    // Negative value means barycentered time series is smaller than topocentric and vice versa.
    // To calculate N_out, we only count the meaningful changes
    int net_shift = 0;
    for (int i = 0 ; i < numdiffbins ; ++i) {
        if (diffbins[i] > 0 && abs(diffbins[i]) < N) {
            net_shift++;
        }
        else if (diffbins[i] < 0 && abs(diffbins[i]) < N) {
            net_shift--;
        }
    }
    std::cout << "Net shift: " << net_shift << std::endl;

    int N_out = N + net_shift;

    // Count bin removals and additions
    int removals = std::ranges::count_if(diffbins, [](int x){ return x < 0; });
    int additions = std::ranges::count_if(diffbins, [](int x){ return x > 0; });
    std::cout << "removals: " << removals << ", additions: " << diffbins.size()-removals << std::endl; 

    // Creat resample map inForOut
    std::vector<int> inForOut(N_out);
    std::vector<int> insertPositions(additions);
    start = std::chrono::high_resolution_clock::now();
    createResampleMap(N, N_out, diffbins.data(), numdiffbins, inForOut.data(), insertPositions.data());
    end = std::chrono::high_resolution_clock::now();
    diff = end-start;
    std::cout << "createResampleMap took " << diff.count() << " seconds\n";


    start = std::chrono::high_resolution_clock::now();
    std::vector<float> barycentered_data(N_out * nchans);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    std::cout << "Initializing barycentered_data took " << diff.count() << " seconds\n";

    // Barycenter timeseries using the resample map inForOut, moving average

    //baryCenterTimeSeries(data.data(), barycentered_data.data(), nchans, N_out, inForOut.data());

    /* resampleTimeSeriesMovAvg(data, barycentered_data, nchans, N_out, 
                               inForOut, additions, insertPositions); */

    resampleTimeSeriesLocAvg(data.data(), barycentered_data.data(), nchans, N_out, 
                               inForOut, insertPositions, 10000);


    std::cout << "Summary" << std::endl;
    std::cout << "numbarybts = " << numbarypts << std::endl;
    std::cout << "numdiffbins = " << numdiffbins << std::endl;


    return 0;
}