#ifndef DEDISP_KERNELS_H_INCLUDE_GUARD
#define DEDISP_KERNELS_H_INCLUDE_GUARD

#include "dedisp_types.h"

#include <vector>

namespace dedisp
{
namespace kernel
{

void generate_delay_table(dedisp_float* h_delay_table, dedisp_size nchans,
                          dedisp_float dt, dedisp_float f0, dedisp_float df);

dedisp_float get_smearing(dedisp_float dt, dedisp_float pulse_width,
                          dedisp_float f0, dedisp_size nchans, dedisp_float df,
                          dedisp_float DM, dedisp_float deltaDM);

void generate_dm_list(std::vector<dedisp_float>& dm_table,
                      dedisp_float dm_start, dedisp_float dm_end,
                      double dt, double ti, double f0, double df,
                      dedisp_size nchans, double tol);

} // end namespace kernels
} // end namespace dedisp

#endif // DEDISP_KERNELS_H_INCLUDE_GUARD