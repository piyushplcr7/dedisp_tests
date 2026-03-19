#include "fil.hpp"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <omp.h>
#include <iostream>
#include "barycenter_presto_utils.h"
#include "matrix_view.hpp"

// -------------------------------------------------------------------------
// Constructor
// -------------------------------------------------------------------------

Fil::Fil(const char* filename){
    filename_ = filename;
    parseHeader();

    // Build ascending frequency array to match the Fits convention:
    //   freqs_[0]       = lowest  = fch1_ + (nchan-1) * foff_
    //   freqs_[nchan-1] = highest = fch1_
    // This ensures:
    //   f0()  = freqs_.back()  = fch1_             (highest, Plan reference)
    //   ddf() = (freqs_[0] - freqs_[nchan-1]) / (nchan-1) = foff_  (negative)
    freqs_.resize(nchans_read_);
    for (int c = 0; c < nchans_read_; ++c) {
        freqs_[c] = fch1_ + (nchans_read_ - 1 - c) * foff_;
    }

    // Build string metadata
    snprintf(telescope_name_, sizeof(telescope_name_), "%s", telescopeName(telescope_id_));
    snprintf(instrument_,     sizeof(instrument_),     "%s", machineName(machine_id_));
    snprintf(observer_,       sizeof(observer_),       "%s", "Unknown");
    snprintf(projid_,         sizeof(projid_),         "%s", "Unknown");
    snprintf(dateobs_,        sizeof(dateobs_),        "MJD%.10f", epoch_);

    // Convert SIGPROC raj/dej to colon-separated strings for TEMPO / barycenter
    rajToString(src_raj_, ra_, sizeof(ra_));
    dejToString(src_dej_, dec_, sizeof(dec_));

    // Derive TEMPO observatory code from telescope name
    telescope_to_tempocode(telescope_name_, outscope_, obs_);
}

// -------------------------------------------------------------------------
// Header parsing
// -------------------------------------------------------------------------

void Fil::parseHeader() {
    FILE* file = fopen(filename_, "r");
    if (!file) {
        fprintf(stderr, "Fil: error opening filterbank file '%s'\n", filename_);
        exit(1);
    }

    char string[80];
    int  nchar;

    for (;;) {
        strcpy(string, "ERROR");
        if (fread(&nchar, sizeof(int), 1, file) != 1) break;

        // Skip implausible key lengths
        if (!(nchar > 1 && nchar < 80)) continue;

        if (fread(string, nchar, 1, file) != 1) break;
        string[nchar] = '\0';

        if (strcmp(string, "HEADER_END") == 0) break;

        // Numeric fields
        if      (strcmp(string, "tsamp")        == 0) fread(&tbin_,        sizeof(double), 1, file);
        else if (strcmp(string, "tstart")       == 0) fread(&epoch_,       sizeof(double), 1, file);
        else if (strcmp(string, "fch1")         == 0) fread(&fch1_,        sizeof(double), 1, file);
        else if (strcmp(string, "foff")         == 0) fread(&foff_,        sizeof(double), 1, file);
        else if (strcmp(string, "nchans")       == 0) fread(&nchans_read_, sizeof(int),    1, file);
        else if (strcmp(string, "nifs")         == 0) fread(&npol_,        sizeof(int),    1, file);
        else if (strcmp(string, "nbits")        == 0) fread(&nbit_,        sizeof(int),    1, file);
        if (nbit_ % 8 != 0 || nbit_ < 8) {
            std::cerr << "Fil: unsupported nbits value " << nbit_ << " (only 8 or multiples of 8 supported)" << std::endl;
            std::abort();
        }
        else if (strcmp(string, "nsamples")     == 0) {
            // Optional pre-declared sample count; we recompute from file size below.
            int dummy; fread(&dummy, sizeof(int), 1, file);
        }
        else if (strcmp(string, "src_raj")      == 0) fread(&src_raj_,     sizeof(double), 1, file);
        else if (strcmp(string, "src_dej")      == 0) fread(&src_dej_,     sizeof(double), 1, file);
        else if (strcmp(string, "telescope_id") == 0) fread(&telescope_id_, sizeof(int),   1, file);
        else if (strcmp(string, "machine_id")   == 0) fread(&machine_id_,  sizeof(int),    1, file);
        // source_name is a length-prefixed string
        else if (strcmp(string, "source_name")  == 0) {
            int slen;
            if (fread(&slen, sizeof(int), 1, file) == 1 && slen > 0 && slen < 80) {
                fread(object_name_, 1, slen, file);
                object_name_[slen] = '\0';
            }
        }
        // az_start / za_start — read and discard
        else if (strcmp(string, "az_start") == 0 || strcmp(string, "za_start") == 0) {
            double dummy; fread(&dummy, sizeof(double), 1, file);
        }
    }

    // Compute header size and derive nsamp from file size
    headersize_ = (size_t)ftell(file);
    fseek(file, 0, SEEK_END);
    file_size_  = (size_t)ftell(file);
    fclose(file);

    buffersize_ = file_size_ - headersize_;

    // Treat the entire file as a single subintegration so that
    // dimTime(downsamp) = dimTime_ / downsamp, and the fitscontainer
    // downsampling check (nsblk % downsamp == 0) maps to (nsamp % downsamp == 0).
    dimTime_ = (int)(buffersize_ / ((size_t)nchans_read_ * npol_ * (nbit_ / 8)));

    file_size_aligned_ = ((file_size_ + 4095) / 4096) * 4096;
}

// -------------------------------------------------------------------------
// Metadata helpers
// -------------------------------------------------------------------------

// Convert SIGPROC src_raj (HHMMSS.SSSSSS) → "HH:MM:SS.SSSSSS"
void Fil::rajToString(double raj, char* out, size_t sz) {
    int    h = (int)(raj / 10000.0);
    int    m = (int)((raj - h * 10000.0) / 100.0);
    double s = raj - h * 10000.0 - m * 100.0;
    snprintf(out, sz, "%02d:%02d:%010.6f", h, m, s);
}

// Convert SIGPROC src_dej (±DDMMSS.SSSSSS) → "±DD:MM:SS.SSSSSS"
void Fil::dejToString(double dej, char* out, size_t sz) {
    char   sign = (dej >= 0.0) ? '+' : '-';
    double adej = fabs(dej);
    int    d    = (int)(adej / 10000.0);
    int    m    = (int)((adej - d * 10000.0) / 100.0);
    double s    = adej - d * 10000.0 - m * 100.0;
    snprintf(out, sz, "%c%02d:%02d:%010.6f", sign, d, m, s);
}

const char* Fil::telescopeName(int id) {
    switch (id) {
        case 0:  return "Fake";
        case 1:  return "Arecibo";
        case 2:  return "Ooty";
        case 3:  return "Nancay";
        case 4:  return "Parkes";
        case 5:  return "Jodrell";
        case 6:  return "GBT";
        case 7:  return "GMRT";
        case 8:  return "Effelsberg";
        case 9:  return "ATA";
        case 10: return "UTR-2";
        case 11: return "LOFAR";
        default: return "Unknown";
    }
}

const char* Fil::machineName(int id) {
    switch (id) {
        case 0:  return "FAKE";
        case 1:  return "PSPM";
        case 2:  return "WAPP";
        case 3:  return "AOFTM";
        case 4:  return "BPP";
        case 5:  return "OOTY";
        case 6:  return "SCAMP";
        case 7:  return "SPIGOT";
        case 10: return "ARTEMIS";
        case 11: return "Cobalt";
        default: return "Unknown";
    }
}

// -------------------------------------------------------------------------
// printInfo
// -------------------------------------------------------------------------

void Fil::printInfo() const noexcept {
    printf("------------------------------ FIL INFO  "
           "----------------------------\n");
    printf("filename         = %s\n",   filename_);
    printf("source_name      = %s\n",   object_name_);
    printf("telescope        = %s (id=%d)\n", telescope_name_, telescope_id_);
    printf("machine          = %s (id=%d)\n", instrument_, machine_id_);
    printf("ra               = %s\n",   ra_);
    printf("dec              = %s\n",   dec_);
    printf("obs (TEMPO)      = %s\n",   obs_);
    printf("epoch (tstart)   = %.15f\n", epoch_);
    printf("tbin (tsamp)     = %.15e s\n", tbin_);
    printf("fch1             = %.15f MHz\n", fch1_);
    printf("foff             = %.15f MHz\n", foff_);
    printf("nchans           = %d\n",   nchans_read_);
    printf("nbit             = %d\n",   nbit_);
    printf("nif (npol)       = %d\n",   npol_);
    printf("nsamp (nsblk)    = %d\n",   dimTime_);
    printf("headersize       = %zu bytes\n", headersize_);
    printf("buffersize       = %zu bytes\n", buffersize_);
    printf("file_size        = %zu bytes\n", file_size_);
    printf("lo freq          = %.6f MHz\n",  freqs_.front());
    printf("hi freq          = %.6f MHz\n",  freqs_.back());
    printf("bw               = %.6f MHz\n",  bw());
    printf("ddf              = %.15f MHz\n", ddf());
    printf("ephem            = %s\n",   ephem_);
}

// -------------------------------------------------------------------------
// I/O: write data back to disk as a SIGPROC filterbank file
// -------------------------------------------------------------------------

void Fil::writeToDisk(const char* outfile) const {
    FILE* fp = fopen(outfile, "wb");
    if (!fp) {
        fprintf(stderr, "Fil::writeToDisk: cannot open '%s' for writing\n", outfile);
        return;
    }

    const unsigned char* data_start = aligned_filesize_buffer_ + headersize_;
    fwrite(data_start, 1, buffersize_, fp);

    fclose(fp);
    printf("Fil::writeToDisk: wrote %s (%zu bytes)\n", outfile, buffersize_);
}

// -------------------------------------------------------------------------
// I/O: direct-IO read, mirrors Fits::getDataFromRows / extractDataDirect
// -------------------------------------------------------------------------


// -------------------------------------------------------------------------
// Data reduction helpers
// -------------------------------------------------------------------------

// Convert raw 8-bit filterbank samples to float (no downsampling).
// Raw layout:  raw[t * nchan + c],  c=0 → fch1 (highest freq, SIGPROC order).
// Output layout: identical channel order — channel 0 remains the highest freq
// channel, consistent with what Plan expects (freq[0] = f0 = fch1).
// Assumption: nbit = 8, nif = 1 (poln index ignored, must be 0).
void Fil::reduceRawData(const unsigned char* raw, float* data,
                        int nchan, size_t nsamp) {
#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < nsamp; ++t) {
        const unsigned char* row = raw + t * (size_t)nchan;
        float*               out = data + t * (size_t)nchan;
        for (int c = 0; c < nchan; ++c) {
            out[c] = (float)row[c];
        }
    }
}

// Same as reduceRawData but sums every `downsamp` consecutive time samples.
// The caller is responsible for the data_ buffer being zero-initialised
// (make_unique<float[]> guarantees this in fitscontainer / filecontainer).
// Requires: nsamp % downsamp == 0.
void Fil::reduceRawDataDownSamp(const unsigned char* raw,
                                float* data,
                                int nchan,
                                size_t nsamp,
                                unsigned int downsamp)
{
#pragma omp parallel for schedule(static)
    for (size_t t_out = 0; t_out < nsamp / downsamp; ++t_out) {
        float* out = data + t_out * (size_t)nchan;

        for (size_t k = 0; k < downsamp; ++k) {
            size_t t = t_out * downsamp + k;
            const unsigned char* row = raw + t * (size_t)nchan;

            for (int c = 0; c < nchan; ++c) {
                out[c] += (float)row[c];
            }
        }
    }
}

void Fil::reduceRawData8(const unsigned char* raw, unsigned char* data,
                        int nchan, size_t nsamp) {
#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < nsamp; ++t) {
        const unsigned char* row = raw + t * (size_t)nchan;
        unsigned char*     out = data + t * (size_t)nchan;
        for (int c = 0; c < nchan; ++c) {
            out[c] = row[c];
        }
    }
}

void Fil::reduceRawDataDownSamp8(const unsigned char* raw,
                                 unsigned char* data,
                                 int nchan,
                                 size_t nsamp,
                                 unsigned int downsamp) {
    #pragma omp parallel
    {
        std::vector<int> acc(nchan);

    #pragma omp for schedule(static)
        for (size_t t_out = 0; t_out < nsamp / downsamp; ++t_out) {
            unsigned char* out = data + t_out * (size_t)nchan;

            std::fill(acc.begin(), acc.end(), 0);

            for (size_t k = 0; k < downsamp; ++k) {
                size_t t = t_out * downsamp + k;
                const unsigned char* row = raw + t * (size_t)nchan;

                for (int c = 0; c < nchan; ++c) {
                    acc[c] += row[c];
                }
            }

            for (int c = 0; c < nchan; ++c) {
                int val = (acc[c] + downsamp / 2) / downsamp;
                if (val > 255) val = 255;
                out[c] = static_cast<unsigned char>(val);
            }
        }
    }
}

// -------------------------------------------------------------------------
// reduceData — public entry point, mirrors Fits::reduceData
// -------------------------------------------------------------------------

void Fil::reduceData(unsigned char* outBuf, int out_bits, int poln, unsigned int downsamp) {
#ifdef TESTDEDISP_DEBUG
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    const unsigned char* raw = aligned_filesize_buffer_ + headersize_;
    size_t nsamp = (size_t)dimTime_;

    if (out_bits == 32) {
        float* data = reinterpret_cast<float*>(outBuf);

        if (downsamp == 1) {
            reduceRawData(raw, data, nchans_read_, nsamp);
        } else {
            reduceRawDataDownSamp(raw, data, nchans_read_, nsamp, downsamp);

            size_t reduced_len = getNumElements(downsamp);
    #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < reduced_len; ++i) {
                data[i] /= downsamp;
            }
        }
    }
    else if (out_bits == 8) {
        unsigned char* data = outBuf;
        if (downsamp == 1) {
            reduceRawData8(raw, data, nchans_read_, nsamp);
        } else {
            reduceRawDataDownSamp8(raw, data, nchans_read_, nsamp, downsamp);
        }
    }
    else {
        std::cerr << "Unsupported out_bits value!" << std::endl;
        std::abort();
    }
    

#ifdef TESTDEDISP_DEBUG
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(
                         end_time - start_time).count();
    std::cout << "Fil::reduceData finished, time: "
              << (double)duration_us / 1e6 << " s" << std::endl;
#endif
}
