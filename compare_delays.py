import numpy as np
import sys
#import matplotlib.pyplot as plt

file1 = sys.argv[1]
file2 = sys.argv[2]

# Load as float32 arrays (adjust dtype if your files are float64 etc.)
arr1 = np.loadtxt(file1, dtype=np.int64)
arr2 = np.loadtxt(file2, dtype=np.int64)

print(f"File1: {file1}, length: {len(arr1)}")
print(f"File2: {file2}, length: {len(arr2)}")

# Find points where they differ
n = min(len(arr1), len(arr2))

print(f"Comparing first {n} points from each file...")
subset1 = arr1[:n]
subset2 = arr2[:n]  
differences = subset1 - subset2
errors = np.abs(differences)


# Find non zero error indices
nonzero_indices = np.where(errors != 0)[0]
print(f"Number of differing points: {len(nonzero_indices)}")
print(f"Indices of differing points: {nonzero_indices}")
print(f"Values at differing points in file1: {subset1[nonzero_indices]}")
print(f"Values at differing points in file2: {subset2[nonzero_indices]}")
print(f"Differences at differing points: {differences[nonzero_indices]}")