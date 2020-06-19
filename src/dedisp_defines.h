#ifndef DEDISP_DEFINES_H_INCLUDE_GUARD
#define DEDISP_DEFINES_H_INCLUDE_GUARD

#define DEDISP_DEFAULT_GULP_SIZE 65536 //131072

// TODO: Make sure this doesn't limit GPU constant memory
//         available to users.
#define DEDISP_MAX_NCHANS 8192

// Internal word type used for transpose and dedispersion kernel
typedef unsigned int dedisp_word;

#endif // DEDISP_DEFINES_H_INCLUDE_GUARD

// Kernel tuning parameters
#define DEDISP_SAMPS_PER_THREAD 2 // 4 is better for Fermi?