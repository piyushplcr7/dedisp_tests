#!/usr/bin/env python3
"""
Verify that a modified FITS file has the expected scales, offsets, and weights
for the GPU-matching 8-bit unpack formula, and that the raw data is unchanged.

Expected values (for nchan channels):
    DAT_SCL  = 1.0 / nchan   (per channel, per polarisation)
    DAT_OFFS = -127.5 / nchan (per channel, per polarisation)
    DAT_WTS  = 1.0            (per channel)

Usage:
    python verify_modified_fits.py <original.fits> <modified.fits> [--num-rows N]
    python verify_modified_fits.py <dir_original> <dir_modified> [--num-rows N]
"""

import argparse
import os
import sys

import numpy as np
from astropy.io import fits


def verify_pair(orig_path, mod_path, num_rows):
    """Verify a single original/modified FITS pair. Returns True if all checks pass."""
    hdul_orig = fits.open(orig_path)
    hdul_mod = fits.open(mod_path)

    hdr = hdul_mod[1].header
    nchan = hdr["NCHAN"]
    npol = hdr["NPOL"]

    orig_data = hdul_orig[1].data
    mod_data = hdul_mod[1].data

    nrows = min(num_rows, orig_data.shape[0], mod_data.shape[0])

    expected_scl = np.float32(1.0 / nchan)
    expected_offs = np.float32(-127.5 / nchan)
    expected_wts = np.float32(1.0)

    print(f"  nchan={nchan}, npol={npol}, nrows={nrows}")
    print(f"  Expected: SCL={expected_scl}, OFFS={expected_offs}, WTS={expected_wts}")

    all_ok = True

    # --- DAT_WTS ---
    wts = mod_data["DAT_WTS"][:nrows, :]
    wts_diff = np.max(np.abs(wts - expected_wts))
    wts_ok = wts_diff == 0
    print(f"  DAT_WTS:  max diff from {expected_wts} = {wts_diff}  {'OK' if wts_ok else 'FAIL'}")
    if not wts_ok:
        all_ok = False

    # --- DAT_SCL (check each polarisation) ---
    scl = mod_data["DAT_SCL"][:nrows, :]
    for p in range(npol):
        scl_p = scl[:, nchan * p : nchan * (p + 1)]
        scl_diff = np.max(np.abs(scl_p - expected_scl))
        scl_ok = scl_diff == 0
        print(f"  DAT_SCL[poln={p}]:  max diff from {expected_scl} = {scl_diff}  {'OK' if scl_ok else 'FAIL'}")
        if not scl_ok:
            all_ok = False

    # --- DAT_OFFS (check each polarisation) ---
    offs = mod_data["DAT_OFFS"][:nrows, :]
    for p in range(npol):
        offs_p = offs[:, nchan * p : nchan * (p + 1)]
        offs_diff = np.max(np.abs(offs_p - expected_offs))
        offs_ok = offs_diff == 0
        print(f"  DAT_OFFS[poln={p}]: max diff from {expected_offs} = {offs_diff}  {'OK' if offs_ok else 'FAIL'}")
        if not offs_ok:
            all_ok = False

    # --- DATA bytes identical ---
    orig_bytes = orig_data["DATA"][:nrows]
    mod_bytes = mod_data["DATA"][:nrows]
    bytes_diff = np.max(np.abs(orig_bytes.astype(np.int16) - mod_bytes.astype(np.int16)))
    bytes_match = bytes_diff == 0
    print(f"  DATA:     max byte diff = {bytes_diff}  {'OK' if bytes_match else 'FAIL'}")
    if not bytes_match:
        nonzero = np.sum(np.abs(orig_bytes.astype(np.int16) - mod_bytes.astype(np.int16)) > 0)
        print(f"            nonzero pixels = {nonzero}/{orig_bytes.size}")
        all_ok = False

    hdul_orig.close()
    hdul_mod.close()

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Verify modified FITS files for GPU-matching unpack.")
    parser.add_argument("original", help="Original FITS file or directory")
    parser.add_argument("modified", help="Modified FITS file or directory")
    parser.add_argument("--num-rows", type=int, default=0, help="Max rows to check (0 = all)")
    args = parser.parse_args()

    num_rows = args.num_rows if args.num_rows > 0 else 999999999

    # Single file mode
    if os.path.isfile(args.original) and os.path.isfile(args.modified):
        print(f"--- {os.path.basename(args.original)} vs {os.path.basename(args.modified)} ---")
        ok = verify_pair(args.original, args.modified, num_rows)
        print(f"\nResult: {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    # Directory mode
    if not (os.path.isdir(args.original) and os.path.isdir(args.modified)):
        print("ERROR: Both arguments must be files or both must be directories.")
        sys.exit(1)

    orig_files = {f for f in os.listdir(args.original) if f.lower().endswith(".fits")}
    mod_files = {f for f in os.listdir(args.modified) if f.lower().endswith(".fits")}

    # Match modified files back to originals (strip _modified suffix)
    pairs = []
    for mf in sorted(mod_files):
        base = mf.replace("_modified", "")
        if base in orig_files:
            pairs.append((base, mf))
        elif mf in orig_files:
            pairs.append((mf, mf))

    if not pairs:
        print("No matching FITS file pairs found.")
        sys.exit(1)

    print(f"Found {len(pairs)} matching pair(s).\n")

    all_pass = True
    for orig_name, mod_name in pairs:
        print(f"--- {orig_name} vs {mod_name} ---")
        ok = verify_pair(
            os.path.join(args.original, orig_name),
            os.path.join(args.modified, mod_name),
            num_rows,
        )
        status = "PASS" if ok else "FAIL"
        print(f"  => {status}\n")
        if not ok:
            all_pass = False

    print("=" * 50)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
