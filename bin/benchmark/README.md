# bench\<x\>
Performance benchmarking implementation for the dedisp library with different implementations of dedisp plan.
Using parameterizable settings for the dedisp plan.

Adapted from the original dedisp test application from Paul Ray (2013).

Note: these applications are intended for performance benchmarking.
The benchmarking applications work with **unitialized data** and do not produce a valid result for further analysis.
Performance is **not affected by the contents of the data** and as most of the test time is spent on creating and allocating valid input data and checking the contents of the output this is a sensible optimization for performance benchmarking.
The **parameters used** for the dedisp plan (e.g. nr DMs, nr of samples, etc.) **do affect the performance** and are used in the benchmarking applications.
The benchmarking applications run best on a system with at least 128 GB RAM.

## Usage
- `<x>` can be one of current three implementations for dedispersion: `dedisp`, `tdd` or `fdd`
- For representative results one should run multiple iterations of the benchmarks and divide the accumulated reported times by the number of iterations
    - e.g. use argument `-i 10` for 10 iterations
- On a multi-socket CPU system the application should be bound to the socket that is connected to the used GPU, use numactl for nodebinding, e.g. to node 0 with `numactl --cpunodebind=0`
- Environment variables for the configuration of implementation alternatives or experimental features for `fdd` might be set at the time of executing the application, refer to the [Documentation](../../Documentation.md) and `fdd` source files for more information on these features.

## Arguments
- `-n [ntrials]`        Number of DM ntrials (default: 100)
- `-s [samples]`        Number of samples to generate (default: 120000)
- `-c [samples]`        Number of channels to generate (default: 1024)
- `-r [DM start(,end)]` Alternative Start (default: 0) (and optional end) values of DM range to dedisperse (use end instead of -n)
- `-t [Tobs]`           Alternative observation time to use [seconds] (use instead of -s)
- `-i [niterations]`    Number of iterations for the dedipsersion plan and execution (default: 1)
- `-q`                  Quiet; no verbose information to screen (default: verbose reporting)
- `-h`                  This help

## Example with default settings
- `numactl --cpunodebind=0 <install dir>/dedisp/bin/bench<dedisp, tdd or fdd>`

Uses the following data:
```
------------------------ INPUT AND OUTPUT DATA ---------------------------
Frequency of highest chanel [MHz]            : 1581.0000
NCHANS (Channel Width [MHz])                 : 1024 (-0.390625)
Sample time (after downsampling by 1 [us])   : 64.000000
Observation duration [s]                     : 7.680000 (120000 samples)
Input data                                   :  8 bits
Output data                                  : 32 bits
Input data array size                        : 468 MB
Output (max) data array size                 : 45 MB
Shared input/output dummy buffer size        : 468 MB, 0 GB

----------------------------- DM COMPUTATIONS  ----------------------------
Computing 100 DMs from 0.000000 to 198.000000 pc/cm^3
Max DM delay is 4063 samples (0 seconds)
Computing 115937 out of 120000 total samples (96.61% efficiency)

```

## Example with settings for 5 minutes of input and 1024 trial-DMs
- `numactl --cpunodebind=0 <install dir>/dedisp/bin/bench<dedisp, tdd or fdd> -s 4689920 -n 1024`

Uses the following data:
```
------------------------ INPUT AND OUTPUT DATA ---------------------------
Frequency of highest chanel [MHz]            : 1581.0000
NCHANS (Channel Width [MHz])                 : 1024 (-0.390625)
Sample time (after downsampling by 1 [us])   : 64.000000
Observation duration [s]                     : 300.154877 (4689920 samples)
Input data                                   :  8 bits
Output data                                  : 32 bits
Input data array size                        : 18320 MB
Output (max) data array size                 : 18320 MB
Shared input/output dummy buffer size        : 18320 MB, 17 GB

----------------------------- DM COMPUTATIONS  ----------------------------
Computing 1024 DMs from 0.000000 to 2046.000000 pc/cm^3
Max DM delay is 41980 samples (3 seconds)
Computing 4647940 out of 4689920 total samples (99.10% efficiency)
```

## Benchmark automation and analysis scripts
One can also use the available python scripts (in this repository at: `dedisp/python`) for benchmarking and analysis of multiple iterations over a larger parameter space. For example:
- Test the benchmark script: `./<install dir>/dedisp/bin/python/run_benchmarks.py <install dir>/dedisp/bin --dryRun`
- Run the actual benchmarks: `./<install dir>/dedisp/bin/python/run_benchmarks.py <install dir>/dedisp/bin`
- This script creates the benchmark results directory with result files per benchmark: `./bench_results_<date-time>`
- Analyze the output from the benchmarks: `./<install dir>/dedisp/bin/python/run_benchmarks_analysis.py <path to benchmark results directory>`