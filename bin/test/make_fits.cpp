// make_fits.cpp
//
// Generate a fake PSRFITS search-mode file matching the structure that
// src/dataHandlers/fits/fits.cpp expects (the G0057_* MWA files).
//
// The header is a hardcoded template of the real file; dimensions and
// observation metadata are CLI-overridable, and the DATA block is filled
// by one of several injection modes (noise / pulse / periodic / impulse /
// constant). DAT_WTS/DAT_OFFS/DAT_SCL default to identity (wt=1, scl=1,
// offs=0) so the reduced output equals the injected byte exactly.
//
// Standalone build (no project build chain needed):
//   g++ -O2 -std=c++17 make_fits.cpp -o make_fits -lcfitsio
//
// Example:
//   ./make_fits --out fake.fits --mode pulse --dm 50 --nchan 256 --npol 1 \
//               --nsblk 1024 --nsubint 8 --amp 100 --baseline 30
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>
#include <random>
#include "fitsio.h"

// ---------------------------------------------------------------------------
// Dispersion constant (PRESTO / standard): delay[s] = K * DM * (f^-2 - fref^-2)
// with frequencies in MHz.  K = 1/2.41e-4 s MHz^2 cm^3 / pc = 4148.808 ...
// ---------------------------------------------------------------------------
static const double DM_CONST = 4148.808; // MHz^2 pc^-1 cm^3 s

struct Config {
  // Output
  std::string out = "fake.fits";

  // Dimensions
  int    nchan   = 256;     // NCHAN
  int    npol    = 1;       // NPOL  (reader selects poln; IQUV uses 4)
  int    nsblk   = 1024;    // NSBLK  (spectra per subint / row)
  int    nsubint = 8;       // NAXIS2 (number of rows)
  int    nbin    = 1;       // NBIN   (search mode -> 1)
  double tbin    = 1e-4;    // [s] time per sample

  // Frequency axis: freq[i] = fch1 + i*foff  (MHz).
  // Defaults reproduce the real file's centre/bandwidth (154.24 MHz, 30.72 MHz).
  double fch1 = 154.24 - 30.72 / 2.0 + 0.01 / 2.0; // first channel centre
  double foff = 0.01;                              // channel width

  // Injection
  std::string mode = "noise"; // noise|pulse|periodic|impulse|constant
  double dm       = 50.0;     // [pc/cm^3]   (pulse/periodic)
  double t0       = -1.0;     // [s] pulse arrival at reference freq; <0 => auto
  double width    = 5e-3;     // [s] gaussian temporal sigma of the pulse
  double amp      = 100.0;    // peak amplitude added on top of baseline
  double period   = 0.5;      // [s] pulse period (periodic)
  double baseline = 30.0;     // constant background level (DC offset)
  double noise_mean = 128.0;  // gaussian noise mean (mode=noise)
  double noise_std  = 18.0;   // gaussian noise std  (mode=noise)
  double value    = 200.0;    // fill value (mode=constant) / spike value (impulse)
  long   seed     = 1;

  // Metadata (hardcoded template defaults from the real G0057 file)
  std::string telescope = "MWA";
  std::string backend   = "v5.1.8_85efc14";
  std::string src_name  = "fakesrc";
  std::string ra        = "00:30:27.42";
  std::string dec       = "+04:51:39.73";
  std::string observer  = "MWA User";
  std::string projid    = "G0057";
  std::string dateobs   = "2019-10-18T14:28:06";
  int    stt_imjd = 58774;
  int    stt_smjd = 52521;
  double stt_offs = 0.999999624677002;
  double be_delay = 0.0;
};

static double clip255(double v) {
  if (v < 0.0)   return 0.0;
  if (v > 255.0) return 255.0;
  return v;
}

static void usage(const char* prog) {
  std::printf(
"Usage: %s [options]\n"
"  --out FILE           output filename (default fake.fits)\n"
"  --mode M             noise|pulse|periodic|impulse|constant (default noise)\n"
"  --nchan N            number of channels (default 256)\n"
"  --npol N             number of polarizations (default 1)\n"
"  --nsblk N            spectra per subint/row (default 1024)\n"
"  --nsubint N          number of rows/subints (default 8)\n"
"  --tbin S             seconds per sample (default 1e-4)\n"
"  --fch1 MHz           centre freq of channel 0 (default ~138.885)\n"
"  --foff MHz           channel width, may be negative (default 0.01)\n"
"  --dm D               dispersion measure pc/cm^3 (pulse/periodic, default 50)\n"
"  --t0 S               pulse time at ref freq; <0 = auto-centre (default auto)\n"
"  --width S            gaussian temporal sigma of pulse (default 5e-3)\n"
"  --amp A              pulse peak added on baseline (default 100)\n"
"  --period S           pulse period for periodic mode (default 0.5)\n"
"  --baseline B         DC background level (default 30)\n"
"  --noise-mean M       gaussian noise mean, mode=noise (default 128)\n"
"  --noise-std S        gaussian noise std,  mode=noise (default 18)\n"
"  --value V            fill value (constant) / spike (impulse) (default 200)\n"
"  --seed N             RNG seed (default 1)\n"
"  --src NAME --ra HH:MM:SS --dec +DD:MM:SS --telescope NAME --projid P\n"
"  --help\n", prog);
}

static bool argEq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int main(int argc, char** argv) {
  Config c;

  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    auto next = [&](const char* name) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", name); std::exit(1); }
      return argv[++i];
    };
    if      (argEq(a, "--help")) { usage(argv[0]); return 0; }
    else if (argEq(a, "--out"))        c.out = next(a);
    else if (argEq(a, "--mode"))       c.mode = next(a);
    else if (argEq(a, "--nchan"))      c.nchan = std::atoi(next(a));
    else if (argEq(a, "--npol"))       c.npol = std::atoi(next(a));
    else if (argEq(a, "--nsblk"))      c.nsblk = std::atoi(next(a));
    else if (argEq(a, "--nsubint"))    c.nsubint = std::atoi(next(a));
    else if (argEq(a, "--tbin"))       c.tbin = std::atof(next(a));
    else if (argEq(a, "--fch1"))       c.fch1 = std::atof(next(a));
    else if (argEq(a, "--foff"))       c.foff = std::atof(next(a));
    else if (argEq(a, "--dm"))         c.dm = std::atof(next(a));
    else if (argEq(a, "--t0"))         c.t0 = std::atof(next(a));
    else if (argEq(a, "--width"))      c.width = std::atof(next(a));
    else if (argEq(a, "--amp"))        c.amp = std::atof(next(a));
    else if (argEq(a, "--period"))     c.period = std::atof(next(a));
    else if (argEq(a, "--baseline"))   c.baseline = std::atof(next(a));
    else if (argEq(a, "--noise-mean")) c.noise_mean = std::atof(next(a));
    else if (argEq(a, "--noise-std"))  c.noise_std = std::atof(next(a));
    else if (argEq(a, "--value"))      c.value = std::atof(next(a));
    else if (argEq(a, "--seed"))       c.seed = std::atol(next(a));
    else if (argEq(a, "--src"))        c.src_name = next(a);
    else if (argEq(a, "--ra"))         c.ra = next(a);
    else if (argEq(a, "--dec"))        c.dec = next(a);
    else if (argEq(a, "--telescope"))  c.telescope = next(a);
    else if (argEq(a, "--projid"))     c.projid = next(a);
    else { std::fprintf(stderr, "unknown arg: %s\n", a); usage(argv[0]); return 1; }
  }

  // Derived sizes
  const long   nchan      = c.nchan;
  const long   npol       = c.npol;
  const long   nsblk      = c.nsblk;
  const long   nsubint    = c.nsubint;
  const long   scal_width = nchan * npol;            // DAT_OFFS / DAT_SCL length
  const long   data_len   = (long)c.nbin * nchan * npol * nsblk; // bytes per row
  const long   ntime      = nsblk * nsubint;         // total samples
  const double tbin       = c.tbin;

  // Channel frequencies and reference (top of band -> minimum delay).
  std::vector<float> freqs(nchan);
  double fref = -1e30;
  for (long ch = 0; ch < nchan; ++ch) {
    double f = c.fch1 + (double)ch * c.foff;
    freqs[ch] = (float)f;
    if (f > fref) fref = f;
  }
  // Auto-centre the (single) pulse in time if t0 < 0.
  double t0 = c.t0;
  if (t0 < 0.0) t0 = 0.5 * ntime * tbin;

  // Per-channel dispersion delay [s] relative to reference frequency.
  std::vector<double> delay(nchan);
  for (long ch = 0; ch < nchan; ++ch) {
    double f = freqs[ch];
    delay[ch] = DM_CONST * c.dm * (1.0 / (f * f) - 1.0 / (fref * fref));
  }

  // ----------------------------------------------------------------------
  // Create the FITS file: empty primary HDU + SUBINT binary table.
  // ----------------------------------------------------------------------
  std::remove(c.out.c_str());
  fitsfile* fp = nullptr;
  int status = 0;
  if (fits_create_file(&fp, c.out.c_str(), &status)) { fits_report_error(stderr, status); return 1; }

  // Primary HDU (minimal image, NAXIS=0) + descriptive header.
  long naxes0[1] = {0};
  fits_create_img(fp, BYTE_IMG, 0, naxes0, &status);
  {
    char buf[256];
    fits_write_key_str(fp, "HDRVER",   (char*)"3.4", "Header version", &status);
    fits_write_key_str(fp, "FITSTYPE", (char*)"PSRFITS", "FITS definition for pulsar data files", &status);
    fits_write_key_str(fp, "OBSERVER", (char*)c.observer.c_str(), "Observer name(s)", &status);
    fits_write_key_str(fp, "PROJID",   (char*)c.projid.c_str(), "Project name", &status);
    fits_write_key_str(fp, "TELESCOP", (char*)c.telescope.c_str(), "Telescope name", &status);
    fits_write_key_str(fp, "BACKEND",  (char*)c.backend.c_str(), "Backend ID", &status);
    double bedelay = c.be_delay;
    fits_write_key_dbl(fp, "BE_DELAY", bedelay, 10, "[s] Backend propn delay", &status);
    fits_write_key_str(fp, "OBS_MODE", (char*)"SEARCH", "(PSR, CAL, SEARCH)", &status);
    fits_write_key_str(fp, "DATE-OBS", (char*)c.dateobs.c_str(), "Date of observation", &status);
    fits_write_key_str(fp, "SRC_NAME", (char*)c.src_name.c_str(), "Source or scan ID", &status);
    fits_write_key_str(fp, "COORD_MD", (char*)"J2000", "Coordinate mode", &status);
    fits_write_key_str(fp, "RA",       (char*)c.ra.c_str(), "Right ascension (hh:mm:ss.ssss)", &status);
    fits_write_key_str(fp, "DEC",      (char*)c.dec.c_str(), "Declination (-dd:mm:ss.sss)", &status);
    long imjd = c.stt_imjd, smjd = c.stt_smjd;
    fits_write_key_lng(fp, "STT_IMJD", imjd, "Start MJD (UTC days)", &status);
    fits_write_key_lng(fp, "STT_SMJD", smjd, "[s] Start time (sec past UTC 00h)", &status);
    fits_write_key_dbl(fp, "STT_OFFS", c.stt_offs, 15, "[s] Start time offset", &status);
    (void)buf;
  }

  // SUBINT binary table -------------------------------------------------
  const int tfields = 17;
  // Column TFORMs that depend on dimensions.
  char tf_freq[32], tf_wts[32], tf_offs[32], tf_scl[32], tf_data[32];
  std::snprintf(tf_freq, sizeof tf_freq, "%ldE", nchan);
  std::snprintf(tf_wts,  sizeof tf_wts,  "%ldE", nchan);
  std::snprintf(tf_offs, sizeof tf_offs, "%ldE", scal_width);
  std::snprintf(tf_scl,  sizeof tf_scl,  "%ldE", scal_width);
  std::snprintf(tf_data, sizeof tf_data, "%ldB", data_len);

  const char* ttype[17] = {
    "TSUBINT","OFFS_SUB","LST_SUB","RA_SUB","DEC_SUB","GLON_SUB","GLAT_SUB",
    "FD_ANG","POS_ANG","PAR_ANG","TEL_AZ","TEL_ZEN",
    "DAT_FREQ","DAT_WTS","DAT_OFFS","DAT_SCL","DATA"
  };
  const char* tform[17] = {
    "1D","1D","1D","1D","1D","1D","1D",
    "1E","1E","1E","1E","1E",
    tf_freq, tf_wts, tf_offs, tf_scl, tf_data
  };
  const char* tunit[17] = {
    "s","s","s","deg","deg","deg","deg","deg","deg","deg","deg","deg",
    "MHz","","","","Jy"
  };

  if (fits_create_tbl(fp, BINARY_TBL, nsubint, tfields,
                      (char**)ttype, (char**)tform, (char**)tunit,
                      (char*)"SUBINT", &status)) {
    fits_report_error(stderr, status); return 1;
  }

  // SUBINT header keys the reader (and PSRFITS consumers) look for.
  {
    long lv;
    fits_write_key_str(fp, "INT_TYPE", (char*)"TIME", "Time axis", &status);
    fits_write_key_str(fp, "INT_UNIT", (char*)"SEC", "Unit of time axis", &status);
    fits_write_key_str(fp, "SCALE",    (char*)"FluxDen", "Intensity units", &status);
    lv = npol;  fits_write_key_lng(fp, "NPOL",  lv, "Nr of polarisations", &status);
    fits_write_key_str(fp, "POL_TYPE", (char*)(npol == 4 ? "IQUV" : "AA+BB"), "Polarisation identifier", &status);
    fits_write_key_dbl(fp, "TBIN", tbin, 12, "[s] Time per bin or sample", &status);
    lv = c.nbin; fits_write_key_lng(fp, "NBIN", lv, "Nr of bins", &status);
    long nbits = 8; fits_write_key_lng(fp, "NBITS", nbits, "Nr of bits/datum", &status);
    lv = nchan; fits_write_key_lng(fp, "NCHAN", lv, "Number of channels", &status);
    double chan_bw = c.foff; fits_write_key_dbl(fp, "CHAN_BW", chan_bw, 6, "[MHz] Channel width", &status);
    lv = nsblk; fits_write_key_lng(fp, "NSBLK", lv, "Samples/row", &status);

    // TDIM17 = (NBIN, NCHAN, NPOL, NSBLK) -- fastest axis first.
    char tdim[64];
    std::snprintf(tdim, sizeof tdim, "(%d,%ld,%ld,%ld)", c.nbin, nchan, npol, nsblk);
    fits_write_key_str(fp, "TDIM17", tdim, "Dimensions", &status);
  }
  if (status) { fits_report_error(stderr, status); return 1; }

  // ----------------------------------------------------------------------
  // Per-row column buffers (constant across rows except DATA).
  // ----------------------------------------------------------------------
  std::vector<float>  dat_wts (nchan, 1.0f);        // identity weights
  std::vector<float>  dat_offs(scal_width, 0.0f);   // identity offset
  std::vector<float>  dat_scl (scal_width, 1.0f);   // identity scale
  std::vector<uint8_t> data(data_len);

  std::mt19937_64 rng(c.seed);
  std::normal_distribution<double> gauss(c.noise_mean, c.noise_std);

  std::printf("Generating %s\n", c.out.c_str());
  std::printf("  mode=%s nchan=%ld npol=%ld nsblk=%ld nsubint=%ld tbin=%g\n",
              c.mode.c_str(), nchan, npol, nsblk, nsubint, tbin);
  std::printf("  bytes/row=%ld total samples=%ld duration=%.3fs\n",
              data_len, ntime, ntime * tbin);
  if (c.mode == "pulse" || c.mode == "periodic")
    std::printf("  dm=%g fref=%.4fMHz max delay=%.4fs t0=%.4fs\n",
                c.dm, fref, delay[0], t0);

  for (long sub = 0; sub < nsubint; ++sub) {
    // ---- fill DATA for this subint ----
    if (c.mode == "noise") {
      for (long k = 0; k < data_len; ++k)
        data[k] = (uint8_t)std::lround(clip255(gauss(rng)));
    } else if (c.mode == "constant") {
      uint8_t v = (uint8_t)std::lround(clip255(c.value));
      std::fill(data.begin(), data.end(), v);
    } else {
      // pulse / periodic / impulse : baseline first, then add signal.
      uint8_t base = (uint8_t)std::lround(clip255(c.baseline));
      std::fill(data.begin(), data.end(), base);

      for (long spec = 0; spec < nsblk; ++spec) {
        double t = (double)(sub * nsblk + spec) * tbin; // absolute time of this spectrum
        for (long ch = 0; ch < nchan; ++ch) {
          double sig = 0.0;
          if (c.mode == "pulse") {
            double dt = t - (t0 + delay[ch]);
            sig = c.amp * std::exp(-0.5 * (dt / c.width) * (dt / c.width));
          } else if (c.mode == "periodic") {
            double phase = t - (t0 + delay[ch]);
            // wrap to nearest pulse in [-period/2, period/2]
            double w = phase - c.period * std::floor(phase / c.period + 0.5);
            sig = c.amp * std::exp(-0.5 * (w / c.width) * (w / c.width));
          } else if (c.mode == "impulse") {
            // single bright sample at (t0, all channels along dispersion track)
            long target = (long)std::lround((t0 + delay[ch]) / tbin);
            if (sub * nsblk + spec == target) sig = c.value - c.baseline;
          }
          if (sig != 0.0) {
            double v = (double)base + sig;
            // write to all polarizations identically
            for (long p = 0; p < npol; ++p) {
              long idx = spec * (nchan * npol) + p * nchan + ch; // matches reduceData
              data[idx] = (uint8_t)std::lround(clip255(v));
            }
          }
        }
      }
    }

    long row = sub + 1; // cfitsio rows are 1-based
    fits_write_col(fp, TFLOAT, 13, row, 1, nchan,      freqs.data(),    &status); // DAT_FREQ
    fits_write_col(fp, TFLOAT, 14, row, 1, nchan,      dat_wts.data(),  &status); // DAT_WTS
    fits_write_col(fp, TFLOAT, 15, row, 1, scal_width, dat_offs.data(), &status); // DAT_OFFS
    fits_write_col(fp, TFLOAT, 16, row, 1, scal_width, dat_scl.data(),  &status); // DAT_SCL
    fits_write_col(fp, TBYTE,  17, row, 1, data_len,   data.data(),     &status); // DATA
    // (TSUBINT/OFFS_SUB/... left at zero; the reader does not consume them.)
    if (status) { fits_report_error(stderr, status); return 1; }
  }

  fits_close_file(fp, &status);
  if (status) { fits_report_error(stderr, status); return 1; }
  std::printf("Done: %s\n", c.out.c_str());
  return 0;
}
