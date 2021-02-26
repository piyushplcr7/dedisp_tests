# dedisp
This repository is derived from Ben Barsdell's original GPU De-dedispersion library (code.google.com/p/dedisp)
And forked from https://github.com/ajameson/dedisp. This fork adds an improved (performance) implementation of dedisp,
referred to as Time Domain Dedispersion (TDD) and adds a new dedispersion algorithm for Fourier Domain Dedispersion (FDD).

Installation Instructions:

  1.  Clone the directory
  2.  Set-up cmake in a build directory, e.g. `/build/dedisp`
      * `$ cmake <source dir path> -DCMAKE_INSTALL_PREFIX=<install dir path>`
  3.  Optionally further configure cmake through interactive build settings
      * `$ccmake .`
      * e.g. set `ENABLE_BENCHMARK` to `ON` to enable build for performance reporting [default: ON]
      * e.g. set `DEDISP_DEBUG` to `ON` to enable build with more verbose output (for debugging purposes) [default: OFF]
  4.  make and install
      * `$ make install`

For usage of dedisp test implementations refer to:
* functional test implementations with generated data [README](bin/test/README.md)
* functional test implementations with a filterbank file as input [README](bin/fil/README.md)
* performance test implementations with dummy input [README](bin/benchmark/README.md)

This repository has been developed and tested with cmake 3.16.2, gcc 8.3.0 and CUDA 11.0.1