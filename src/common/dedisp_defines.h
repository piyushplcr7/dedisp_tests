#ifndef DEDISP_DEFINES_H_INCLUDE_GUARD
#define DEDISP_DEFINES_H_INCLUDE_GUARD

// Internal word type used for transpose and dedispersion kernel
typedef unsigned int dedisp_word;

// Kernel tuning parameters
#define DEDISP_SAMPS_PER_THREAD 2 // 4 is better for Fermi?

#endif // DEDISP_DEFINES_H_INCLUDE_GUARD