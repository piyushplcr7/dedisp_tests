#!/usr/bin/env python3
"""
Compare FITS files between two directories.
Directory 1 contains 4-polarisation FITS files.
Directory 2 contains single-polarisation (poln 0) FITS files.
Files are matched by name.

Usage:
    python compare_fits.py <dir_4poln> <dir_1poln> [--output-dir <plot_dir>] [--num-rows <N>]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits


def get_data(hdul, num_rows, poln):
    """Extract weights, offsets, scales, and data bytes from a FITS HDU list."""
    data = hdul[1].data
    nchans = hdul[1].header["NCHAN"]
    npoln = hdul[1].header["NPOL"]

    num_rows = min(num_rows, data.shape[0])

    data_wts = data["DAT_WTS"][:num_rows, :]
    data_offs = data["DAT_OFFS"][:num_rows, nchans * poln : nchans * (poln + 1)]
    data_scls = data["DAT_SCL"][:num_rows, nchans * poln : nchans * (poln + 1)]
    data_bytes = data["DATA"][:num_rows, :, poln, :, 0]

    return data_wts, data_offs, data_scls, data_bytes


def save_heatmap(diff, title, filepath):
    """Save a heatmap of the difference array."""
    plt.figure()
    plt.imshow(diff, cmap="hot", aspect="auto")
    plt.colorbar(label="Difference magnitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()


def compare_file(file_4poln, file_1poln, num_rows, output_dir, basename):
    """Compare a single pair of FITS files and save heatmaps."""
    hdul1 = fits.open(file_4poln)
    hdul2 = fits.open(file_1poln)

    npoln1 = hdul1[1].header["NPOL"]
    npoln2 = hdul2[1].header["NPOL"]
    print(f"  4-poln file NPOL={npoln1}, 1-poln file NPOL={npoln2}")

    wts1, offs1, scls1, bytes1 = get_data(hdul1, num_rows, poln=0)
    wts2, offs2, scls2, bytes2 = get_data(hdul2, num_rows, poln=0)

    hdul1.close()
    hdul2.close()

    diff_wts = np.abs(wts1 - wts2)
    diff_offs = np.abs(offs1 - offs2)
    diff_scls = np.abs(scls1 - scls2)
    diff_bytes = np.abs(bytes1.astype(np.int16) - bytes2.astype(np.int16))

    # Print errors
    print(f"  DAT_WTS  max diff = {np.max(diff_wts)}")
    print(f"  DAT_OFFS max diff = {np.max(diff_offs)}")
    print(f"  DAT_SCL  max diff = {np.max(diff_scls)}")
    print(f"  DATA     max diff = {np.max(diff_bytes)}, "
          f"nonzero pixels = {np.sum(diff_bytes > 0)}/{diff_bytes.size}")

    # Save heatmaps
    prefix = os.path.splitext(basename)[0]
    save_heatmap(diff_wts, f"{prefix} - Weights Diff", os.path.join(output_dir, f"{prefix}_wts.png"))
    save_heatmap(diff_offs, f"{prefix} - Offsets Diff", os.path.join(output_dir, f"{prefix}_offs.png"))
    save_heatmap(diff_scls, f"{prefix} - Scales Diff", os.path.join(output_dir, f"{prefix}_scls.png"))

    # Reshape data diff for 2D heatmap (rows*nsblk, nchans)
    shape = diff_bytes.shape
    diff_bytes_2d = diff_bytes.reshape(shape[0] * shape[1], shape[2])
    save_heatmap(diff_bytes_2d, f"{prefix} - Data Diff", os.path.join(output_dir, f"{prefix}_data.png"))

    return {
        "wts_max": np.max(diff_wts),
        "offs_max": np.max(diff_offs),
        "scls_max": np.max(diff_scls),
        "bytes_max": np.max(diff_bytes),
        "bytes_nonzero": int(np.sum(diff_bytes > 0)),
        "bytes_total": diff_bytes.size,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare 4-poln vs 1-poln FITS files.")
    parser.add_argument("dir_4poln", help="Directory with 4-polarisation FITS files")
    parser.add_argument("dir_1poln", help="Directory with single-polarisation FITS files")
    parser.add_argument("--output-dir", default="compare_plots", help="Directory for heatmap PNGs (default: compare_plots)")
    parser.add_argument("--num-rows", type=int, default=0, help="Max rows to compare per file (0 = all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find matching .fits files
    files_4poln = {f for f in os.listdir(args.dir_4poln) if f.lower().endswith(".fits")}
    files_1poln = {f for f in os.listdir(args.dir_1poln) if f.lower().endswith(".fits")}
    common = sorted(files_4poln & files_1poln)

    if not common:
        print("No matching FITS files found between the two directories.")
        sys.exit(1)

    print(f"Found {len(common)} matching FITS file(s).\n")

    results = {}
    for fname in common:
        print(f"--- {fname} ---")
        path1 = os.path.join(args.dir_4poln, fname)
        path2 = os.path.join(args.dir_1poln, fname)
        num_rows = args.num_rows if args.num_rows > 0 else 999999999
        try:
            results[fname] = compare_file(path1, path2, num_rows, args.output_dir, fname)
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Summary
    if results:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'File':<50} {'wts':>8} {'offs':>10} {'scls':>10} {'data':>8}")
        print("-" * 60)
        for fname, r in results.items():
            print(f"{fname:<50} {r['wts_max']:>8.4f} {r['offs_max']:>10.6f} {r['scls_max']:>10.6f} {r['bytes_max']:>8}")


if __name__ == "__main__":
    main()
