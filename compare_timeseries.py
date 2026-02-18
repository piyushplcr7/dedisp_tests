import numpy as np
import sys
#import matplotlib.pyplot as plt

file1 = sys.argv[1]
file2 = sys.argv[2]

# Load as float32 arrays (adjust dtype if your files are float64 etc.)
arr1 = np.fromfile(file1, dtype=np.float32)
arr2 = np.fromfile(file2, dtype=np.float32)

print(f"File1: {file1}, length: {len(arr1)}")
print(f"File2: {file2}, length: {len(arr2)}")
 
n = min(len(arr1), len(arr2))-1
n = min(n, int(sys.argv[3]) if len(sys.argv) > 3 else n)  # Optionally limit to first N points
print(f"Comparing first {n} points from each file...")
subset1 = arr1[:n]
subset2 = arr2[:n]

# Compute errors
errors = np.abs(subset1 - subset2)

# Find max and min error indices
max_idx = np.argmax(errors)
min_idx = np.argmin(errors)
print("==========================================")
print(f"Max error: {errors[max_idx]} at index {max_idx}")
print(f"  arr1[{max_idx}] = {subset1[max_idx]}")
print(f"  arr2[{max_idx}] = {subset2[max_idx]}")
print("==========================================")
print(f"Min error: {errors[min_idx]} at index {min_idx}")
print(f"  arr1[{min_idx}] = {subset1[min_idx]}")
print(f"  arr2[{min_idx}] = {subset2[min_idx]}")
print("==========================================")
print(f"first few errors: \n {errors[:20]}")
print("==========================================")
""" tol = 1e-2
print(f"Checking errs > {tol}")
print("Errors > tol:")
print("    ",errors[errors > tol])
print("Locations of errors > tol:")
print("    ",np.where(errors > tol))
print("arr1 values:")
print("    ",arr1[np.where(errors > tol)])
print("arr2 values:")
print("    ",arr2[np.where(errors > tol)]) """

# Plot them
""" plt.figure(figsize=(8,5))
plt.plot(subset1, marker='o', label="File 1")
plt.plot(subset2, marker='s', label="File 2")
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("First 20 points from two binary files")
plt.legend()
plt.grid(True)
plt.show() """
