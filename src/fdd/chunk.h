// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#ifndef H_CHUNK_INCLUDE_GUARD
#define H_CHUNK_INCLUDE_GUARD

#include <vector>
#include <cstdio>

#include "helper.h"

namespace dedisp
{

struct Chunk
{
    // Time domain
    unsigned int isamp_start; // scalar index
    unsigned int isamp_end;   // scalar index

    // Frequency domain
    unsigned int ifreq_start; // scalar index
    unsigned int ifreq_end;   // scalar index
    unsigned int nfreq_good;  // number of good samples
};

void compute_chunks(
    unsigned int nsamp,
    unsigned int nsamp_good,
    unsigned int nfft,
    unsigned int nfreq_chunk_padded,
    unsigned int& nfreq_computed,
    std::vector<Chunk>& chunks);

void print_chunks(
    std::vector<Chunk>& chunks);

void copy_chunk_output(
    float* src,
    float* dst,
    unsigned int ndm,
    unsigned int nsamp,
    unsigned int nsamp_computed,
    unsigned int nsamp_padded,
    unsigned int nsamp_good,
    std::vector<Chunk>& chunks);

void generate_spin_frequency_table_chunks(
    std::vector<Chunk>& chunks,
    std::vector<float>& spin_frequencies,
    unsigned int nfreq_chunk,
    unsigned int nfreq_chunk_padded,
    unsigned int nfft,
    float dt);

} // end namespace dedisp

#endif // H_CHUNK_INCLUDE_GUARD