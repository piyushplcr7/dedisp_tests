# Handoff: FDD CPU (OpenMP) port — DM0 output still wrong (input layout)

## RESOLVED 2026-06-10 — root cause was offset/scale, NOT input layout
The leading hypothesis below (polarization interleaving / wrong stride) was
**wrong**. `reduceData` (`src/dataHandlers/fits/fits.cpp:279`) writes the
assembled float buffer as plain time-major `data[t*nchan + chan]`, stride
exactly `nchan` — the CPU `transpose_data` indexing was already correct.

The real bug: the GPU has **two different kernels** and the CPU only matched one.
`transpose_unpack_kernel` (8-bit) applies `(val - 127.5f) * (1/nchan)`, but
`transpose_kernel<float>` (32-bit, `src/fdd/unpack/unpack_kernel.cuh:7-74`) is a
**PURE transpose**: `out = in`, the `scale` arg is ignored, NO offset. The CPU
32-bit branch was wrongly using offset 127.5 / scale nchan, which divides every
spike by ~nchan and pins output to a -127.5 DC floor. The "baseline matched but
spikes vanished" symptom was the tell: GPU's DM0 baseline channel-sum (≈-127.44)
coincidentally ≈ the CPU's -127.5 constant offset.

**Fix (applied):** `FDDCPUPlan::execute_cpu` 32-bit branch now uses
`transpose_data<const float,float>(nchan, nsamp, nchan, nsamp_padded, 0.0f, 1.0f, ...)`.

**To verify (user runs):** rebuild on kuma (`cmake -DGPU_BACKEND=OPENMP`), run
`openmp_test.sh`, then `python3 ~/repos/dedisp_tests/find_shift.py
/scratch/panchal/masterout/out_DM0.00.dat /scratch/panchal/cpuout/out_DM0.00.dat`.
Expect CPU std now in the hundreds with spikes present and matching the GPU
reference to ~1e-3 at DM0.

---
## (original handoff below, kept for context)


## Goal
Finish an **OpenMP-only CPU port** of the FDD (Fourier-Domain Dedispersion) GPU
code so its per-DM `.dat` timeseries match the GPU reference. Backend selected
via `cmake -DGPU_BACKEND=OPENMP`. The CPU path lives in
`src/fdd/FDDCPUPlan.{cpp,hpp}`; the GPU ground truth is
`src/fdd/FDDGPUPlan.cpp`. Driver: `bin/test/testdedisp_omp.cpp` (CPU mirror of
`bin/test/testdedisp_new.cpp`).

Worktree: `/home/panchal/repos/dedisp_tests/.worktrees/nextsilicon` (branch
`nextsilicon`). Build/run happens on SCITAS node **kuma** with module env +
conda `prestoenv_new`. **The user runs ALL builds/scripts/python themselves — do
NOT execute anything; hand them exact commands. Never git commit/add.**

## Current symptom (the blocker)
At **DM=0** the CPU output disagrees with the GPU reference. DM0 is the simplest
case: dedispersion is a pure channel-sum + FFT round-trip (zero phase), so CPU
and GPU MUST be identical here. They are not.

Latest `compare_timeseries.py` (CPU `cpuout` vs GPU `masterout`), both length
1992920:
- Baseline now MATCHES: index 598763 → CPU −127.4410 vs ref −127.4408 (err 6e-4).
- Spikes MISSING: index 1010098 → CPU −127.63 (≈ baseline) vs ref −3642.22.
- "first few errors" ~hundreds.

Before the most recent fix the CPU was totally flat (mean −17, std 1.2). After
the fix the DC is correct (baseline −127.4) but the **high-variance structure
(spikes) is still absent** — CPU reads ~baseline where the GPU reads large
excursions. So the unpack offset/scale is right; the per-sample data the CPU
reads is still wrong → an **input memory-layout / stride bug remains.**

## What is already FIXED and VERIFIED equivalent to the GPU (do not re-litigate)
1. Spin-frequency table `ifreq/nsamp_fft` (not `ifreq/(nsamp*dt)`).
2. Integer-sample delays `roundf(dm*delay)`.
3. FFT sizing: `nsamp_fft=closestOptimal(nsamp+max_delay)`, `nsamp_padded=2*(nsamp_fft/2+1)`, `nfreq=nsamp_fft/2+1`.
4. Zero-pad on the TAIL (`data_nu` at offset 0, tail stays zero) — matches GPU; NOT a front-shift.
5. Output: dirty buffer (`nsamps_computed_=nsamps+max_delay`), `writeOutput` skips first `max_delay`, writes `nsamps-max_delay`. Runs WITHOUT `-cleanout`, mirroring the GPU. Lengths match exactly (1992920), so sizing is correct.
6. Heap-allocated kernel transpose buffers; channels padded to mult-of-32 with zeros.
7. **Input dtype**: the dataLoader's assembled buffer is `std::unique_ptr<float[]> assembledDataBuffer_` (`src/dataHandlers/fitscontainer.hpp:50`) — already-unpacked float. `execute_cpu` now branches on `in_nbits` and reads `const float*` when `in_nbits==32` (was hardcoded `byte_type`, which read float mantissa bytes → the std-1.2 flat output). This fixed the DC but NOT the remaining structure loss.

## Leading hypothesis for the remaining bug
The CPU's `transpose_data` call in `FDDCPUPlan::execute_cpu` uses
`in_stride = nchan` (in floats) and reads `in[t*nchan + chan]`, i.e. it assumes
the float buffer is time-major `[nsamp][nchan]` with exactly `nchan` floats per
time sample. The GPU instead indexes the same buffer via
`nchan_words = nchan/chans_per_word`, `chans_per_word = sizeof(dedisp_word)*8/in_nbits`,
`src_stride = nchan_words*sizeof(dedisp_word)` and a `transpose_unpack` kernel
(`FDDGPUPlan.cpp` ~line 1057-1090 in the MPI path, ~1650-1690 in the simple
path). If the true per-time-sample stride or channel ordering of
`assembledDataBuffer_` differs from `nchan` contiguous floats (e.g. polarization
interleaving — the data is "4 polns, not summed, 8 bit"; `poln=0` is selected
via `ldSeq(chunksize, poln=0)`), the CPU reads across the structure → averages
out the time variation → keeps the right DC but loses the spikes. This matches
the observed symptom precisely.

## Concrete next steps for the fresh agent
1. **Confirm flat vs shifted first.** Have the user run
   `python3 ~/repos/dedisp_tests/find_shift.py <ref> <test>` (prints mean/median/
   **std**/min/max + first-10 + fraction-within-1.0). If CPU std is still ~1-ish
   → still flat (input still being averaged). If std ~hundreds but mismatched →
   a shift/ordering issue instead. This single output bisects the problem.
2. **Pin down the exact layout of `assembledDataBuffer_`**: read how the
   dataLoader fills it (`src/dataHandlers/fitscontainer.{hpp,cpp}` — search
   `assembledDataBuffer_`, `packChannelChunk`, `ldSeq`, `poln`, `downsamp`,
   `nchansLocal`, `nsampsLocal`). Determine floats-per-time-sample and channel
   ordering. Compare to what the GPU's `memcpy2D`/`transpose_unpack` assume
   (`src_stride = nchan_words*sizeof(dedisp_word)`).
3. **Verify runtime values match the GPU**: `nchan`(=m_nchans=nchansLocal),
   `nsamp`(=nsampsLocal), `in_nbits`, `chans_per_word`, `nchan_words`. A
   `TESTDEDISP_DEBUG`-gated print of these + the first ~16 raw floats of `in`
   and the first ~16 `data_nu` values per channel after transpose will show
   immediately whether the CPU transpose is reading the right samples.
4. Fix the CPU `transpose_data` stride/indexing (and possibly add a real unpack
   step) so `data_nu[chan][t]` equals the GPU's post-`transpose_unpack`
   `d_data_t_nu[chan][t]`. Validate at DM0 first (must match to ~1e-4), then a
   few higher DMs.

## Reference run commands (user executes)
GPU reference (`run_master.sh`, uses `dedisp_master` build, `-cleanout`) and CPU
(`openmp_test.sh`, no `-cleanout`) both read `/scratch/panchal/chrisdata/*1.fits`,
`-lodm 0 -dmstep 0.01 -numdms 1000 -multout -nobary`. Outputs in
`/scratch/panchal/{masterout,cpuout}/out_DM0.00.dat`. NOTE: `masterout` is the
**master branch** (`~/repos/dedisp_master`), a different codebase than this
worktree's `FDDGPUPlan.cpp`; the cleanest reference would be this worktree's own
GPU `testdedisp` (its compile errors were fixed this session) but the user is
staying on kuma (CPU only) and not building GPU. Keep that caveat in mind — but
the DM0 flat/structure-loss bug is real regardless of which GPU reference, since
a correct DM0 channel-sum cannot be near-constant.

## Key files
- `src/fdd/FDDCPUPlan.cpp` — `execute_cpu` (transpose at ~line 782, FFT, dedisperse_optimized, IFFT, output copy), `setOutputParams`, `writeOutput`.
- `src/fdd/FDDGPUPlan.cpp` — `execute_gpu` input gulp + `transpose_unpack` (ground truth for input layout).
- `src/fdd/helper.h` — `transpose_data` (out[y][x] = (in[x*in_stride+y]-offset)/scale), `copy_data`.
- `src/dataHandlers/fitscontainer.hpp` — `assembledDataBuffer_` (float[]), `getAssembledDataBfr`, `nbits`, `nchansLocal`, `nsampsLocal`.
- `bin/test/testdedisp_omp.cpp` — CPU driver. `find_shift.py`, `compare_timeseries.py` in repo root.
