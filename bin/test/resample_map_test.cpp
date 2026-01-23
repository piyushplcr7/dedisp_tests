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
    size_t N = 10;
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Number of bin-changes in the topocenteric time series
    int numdiffbins = 3;

    std::vector<int> diffbins = {-3, 5, -7};
    

    printf("idx\t\tdelta\n");
    for (int i  = 0 ; i < numdiffbins ; ++i) {
        printf("%d\t\t%d\n",i , diffbins[i]);
    }

    int meaningful_changes = 0;
    for (int i = 0 ; i < numdiffbins ; ++i) {
        if (diffbins[i] > 0 && abs(diffbins[i]) < N) {
            meaningful_changes++;
        }
        else if (diffbins[i] < 0 && abs(diffbins[i]) < N) {
            meaningful_changes--;
        }
    }
    std::cout << "meaningful changes: " << meaningful_changes << std::endl;

    size_t N_out = N + meaningful_changes;

    // Creat resample map inForOut
    std::vector<int> inForOut(N_out);
    std::vector<int> insertPositions(numdiffbins,-1);
    createResampleMap(N, N_out, diffbins.data(), numdiffbins, inForOut.data(), insertPositions.data());

    for (int i = 0 ; i < inForOut.size(); ++i) {
        std::cout << inForOut[i] << " => " << i << std::endl;
    }

    std::cout << "Insert positions in the output =>" << std::endl;
    for (int i = 0 ; i < numdiffbins ; ++i) {
        if (insertPositions[i] < 0)
            break;
        std::cout << "Insertion at " << insertPositions[i] << " using input val " << inForOut[insertPositions[i]] << std::endl;
    }

    return 0;
}