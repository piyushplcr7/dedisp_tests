# dedisp
This repositry is derived from Ben Barsdell's original GPU De-dedispersion library (code.google.com/p/dedisp)
And forked from https://github.com/ajameson/dedisp

Installation Instructions:

  1.  Clone the directory
  2.  Set-up cmake in a build directory, e.g. `/build/dedisp`
      * `$ cmake <source dir path> -DCMAKE_INSTALL_PREFIX=<install dir path>`
  2.  make and install
      * `$ make install`
  3.  test dedisp
      * `$ make test`

This repository has been developed and tested with cmake 3.16.2, gcc 8.3.0 and CUDA 10.2.89
  
  Usage with `dedisp_fil` from install directory:
  * Run `dedisp` library `$ ./bin/dedisp_fil -f <filterbank file>.fil -r 900 -s 1 -n 100 -o <filter bank file name>`
  * Run python script for analysis (with Anaconda 3 or Python 3 with numpy and matplotlib packages installed): `$ bin/python/plot_burst.py`