#include "common/dedisp_types.h"
#include "common/dedisp_error.hpp"


#define DEDISP_DEFAULT_GULP_SIZE 65536 // 131072

// TODO: Make sure this doesn't limit GPU constant memory
//         available to users.
#define DEDISP_MAX_NCHANS 8192

// Kernel tuning parameters
#define DEDISP_SAMPS_PER_THREAD 2 // 4 is better for Fermi?

void copy_delay_table(
    const void* src,
    size_t count,
    size_t offset,
    cudaStream_t stream);

void copy_killmask(
    const void* src,
    size_t count,
    size_t offset,
    cudaStream_t stream);

bool dedisperse(const dedisp_word*  d_in,
                dedisp_size         in_stride,
                dedisp_size         nsamps,
                dedisp_size         in_nbits,
                dedisp_size         nchans,
                dedisp_size         chan_stride,
                const dedisp_float* d_dm_list,
                dedisp_size         dm_count,
                dedisp_size         dm_stride,
                dedisp_byte*        d_out,
                dedisp_size         out_stride,
                dedisp_size         out_nbits,
                dedisp_size         batch_size,
                dedisp_size         batch_in_stride,
                dedisp_size         batch_dm_stride,
                dedisp_size         batch_chan_stride,
                dedisp_size         batch_out_stride);