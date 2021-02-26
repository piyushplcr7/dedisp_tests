# x_fil
Dedisperse filterbank files with dedisp library with different implementations of dedisp plan.
The code only supports reading 8 bit filterbank files.

Adapted from dedisp_fil repository from Cees Bassa: https://github.com/cbassa/dedisp_fil

## Usage
- Can be used to compare the outcome of different implementations on real data from a `.fil` file
- Or to process `.fil` files directly with the dedisp library
- `x` can be one of current three implementations for dedispersion: `dedisp`, `tdd` or `fdd`
- use with arguments from main `fil` method (refer to `fil.hpp`)

## Arguments
- `-f <FILE file>`      SIGPROC filterbank file to read as input
- `-o <output prefix>`  Output prefix
- `-D [GPU device]`     GPU device number to use (default: 0)
- `-r [DM start(,end)]` Start (and optional end) values of DM range to dedisperse (default: 0,50)
- `-s [DM step]`        Linear DM step to use. (default: optimal DM trials are computed)
- `-n [ntrails]`        Number of DM trails
- `-d [decimate]`       Decimate timeseries by this factor (default: 1)
- `-N [samples]`        Number of samples to write (default: all)
- `-w [pulse width]`    Expected intrinsic pulse width for optimal DM trials (default: 4.0us)
- `-t [tolerance]`      Smearing tolerance factor between DM trials (default: 1.25)
- `-q`                  Quiet; no information to screen
- `-h`                  This help

## Example
- `<install dir>/dedisp/bin/dedisp_fil -f .../data/<file name>.fil -r 900 -s 1 -n 100 -o <output prefix>`