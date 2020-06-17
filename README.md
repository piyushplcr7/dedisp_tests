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
  
  Usage with `dedisp_fil` from install directory:
  * Run `dedisp` library `$ ./bin/dedisp_fil -f <filterbank file>.fil -r 900 -s 1 -n 100 -o <filter bank file name>`
  * Run python script for analysis (with Anaconda 3): `$ bin/plot_burst.py`