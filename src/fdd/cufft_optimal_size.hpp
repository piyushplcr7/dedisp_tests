#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>

// Check if n factors only into the allowed factors (2, 3, 5, 7)
bool isOptimal(size_t n, const std::vector<size_t>& allowedFactors = {2, 3, 5, 7}) {
    // Edge case: values less than 2 are not considered optimal
    if(n < 2)
        return false;
    size_t temp = n;
    // Remove allowed factors
    for (size_t p : allowedFactors) {
        while (temp % p == 0) {
            temp /= p;
        }
    }
    // If nothing remains, then only allowed factors were present.
    return (temp == 1);
}

// Helper function to compute and print the full prime factorization of n.
void printFactorization(size_t n) {
    size_t original = n;
    std::cout << "Prime factorization of " << original << " is: ";
    bool firstFactor = true;
    
    // Factor out 2's.
    size_t count = 0;
    while (n % 2 == 0) {
        count++;
        n /= 2;
    }
    if (count > 0) {
        std::cout << "2^" << count;
        firstFactor = false;
    }
    
    // Factor out odd numbers.
    for (size_t i = 3; i <= static_cast<size_t>(std::sqrt(static_cast<double>(n))); i += 2) {
        count = 0;
        while (n % i == 0) {
            count++;
            n /= i;
        }
        if (count > 0) {
            if (!firstFactor)
                std::cout << " * ";
            std::cout << i;
            if (count > 1)
                std::cout << "^" << count;
            firstFactor = false;
        }
    }
    
    // If n is a prime greater than 2.
    if (n > 2) {
        if (!firstFactor)
            std::cout << " * ";
        std::cout << n;
    }
    std::cout << std::endl;
}

// Finds the closest optimal number (optimal for cuFFT) near n.
// If searchNextLargest is true, searches upward (n, n+1, ...),
// otherwise, searches downward (n, n-1, ...).
size_t closestOptimal(size_t n, bool searchNextLargest = true, const std::vector<size_t>& allowedFactors = {2, 3, 5, 7}) {
    size_t candidate = n;
    
    // Search for an optimal candidate.
    // For downward search, ensure we don't underflow.
    while (!isOptimal(candidate, allowedFactors)) {
        if (searchNextLargest) {
            // Prevent overflow.
            if (candidate == std::numeric_limits<size_t>::max()) {
                std::cerr << "Reached maximum size_t value without finding an optimal candidate." << std::endl;
                break;
            }
            candidate++;
        } else {
            if (candidate == 0) break;
            candidate--;
        }
    }
    
    // Determine the smallest allowed factor.
    size_t minAllowed = *std::min_element(allowedFactors.begin(), allowedFactors.end());
    if (candidate < minAllowed) {
        std::cerr << "Warning: Provided dimension is smaller than the smallest allowed factor. Returning " << minAllowed << std::endl;
        candidate = minAllowed;
    }
    
    // Print the prime factorization for the candidate.
    printFactorization(candidate);
    
    return candidate;
}