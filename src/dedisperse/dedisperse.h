#include "dedisp_types.h"
#include "dedisp_defines.h"
#include "dedisp_error.hpp"

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

unsigned int get_nsamps_per_thread();

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
                cudaStream_t        stream = 0);