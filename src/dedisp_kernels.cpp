#include "dedisp_kernels.hpp"

#include <cstdlib>
#include <cmath>

namespace dedisp
{
namespace kernel
{

void generate_delay_table(dedisp_float* h_delay_table, dedisp_size nchans,
                          dedisp_float dt, dedisp_float f0, dedisp_float df)
{
    for( dedisp_size c=0; c<nchans; ++c ) {
        dedisp_float a = 1.f / (f0+c*df);
        dedisp_float b = 1.f / f0;
        // Note: To higher precision, the constant is 4.148741601e3
        h_delay_table[c] = 4.15e3/dt * (a*a - b*b);
    }
}

dedisp_float get_smearing(dedisp_float dt, dedisp_float pulse_width,
                          dedisp_float f0, dedisp_size nchans, dedisp_float df,
                          dedisp_float DM, dedisp_float deltaDM)
{
    dedisp_float W         = pulse_width;
    dedisp_float BW        = nchans * abs(df);
    dedisp_float fc        = f0 - BW/2;
    dedisp_float inv_fc3   = 1./(fc*fc*fc);
    dedisp_float t_DM      = 8.3*BW*DM*inv_fc3;
    dedisp_float t_deltaDM = 8.3/4*BW*nchans*deltaDM*inv_fc3;
    dedisp_float t_smear   = sqrt(dt*dt + W*W + t_DM*t_DM + t_deltaDM*t_deltaDM);
    return t_smear;
}

void generate_scrunch_list(dedisp_size* scrunch_list,
                           dedisp_size dm_count,
                           dedisp_float dt0,
                           const dedisp_float* dm_list,
                           dedisp_size nchans,
                           dedisp_float f0, dedisp_float df,
                           dedisp_float pulse_width,
                           dedisp_float tol)
{
    // Note: This algorithm always starts with no scrunching and is only
    //         able to 'adapt' the scrunching by doubling in any step.
    // TODO: To improve this it would be nice to allow scrunch_list[0] > 1.
    //         This would probably require changing the output nsamps
    //           according to the mininum scrunch.

    scrunch_list[0] = 1;
    for( dedisp_size d=1; d<dm_count; ++d ) {
        dedisp_float dm = dm_list[d];
        dedisp_float delta_dm = dm - dm_list[d-1];

        dedisp_float smearing = get_smearing(scrunch_list[d-1] * dt0,
                                             pulse_width*1e-6,
                                             f0, nchans, df,
                                             dm, delta_dm);
        dedisp_float smearing2 = get_smearing(scrunch_list[d-1] * 2 * dt0,
                                              pulse_width*1e-6,
                                              f0, nchans, df,
                                              dm, delta_dm);
        if( smearing2 / smearing < tol ) {
            scrunch_list[d] = scrunch_list[d-1] * 2;
        }
        else {
            scrunch_list[d] = scrunch_list[d-1];
        }
    }
}

void generate_dm_list(std::vector<dedisp_float>& dm_table,
                      dedisp_float dm_start, dedisp_float dm_end,
                      double dt, double ti, double f0, double df,
                      dedisp_size nchans, double tol)
{
    // Note: This algorithm originates from Lina Levin
    // Note: Computation done in double precision to match MB's code

    dt *= 1e6;
    double f    = (f0 + ((nchans/2) - 0.5) * df) * 1e-3;
    double tol2 = tol*tol;
    double a    = 8.3 * df / (f*f*f);
    double a2   = a*a;
    double b2   = a2 * (double)(nchans*nchans / 16.0);
    double c    = (dt*dt + ti*ti) * (tol2 - 1.0);

    dm_table.push_back(dm_start);
    while( dm_table.back() < dm_end ) {
        double prev     = dm_table.back();
        double prev2    = prev*prev;
        double k        = c + tol2*a2*prev2;
        double dm = ((b2*prev + sqrt(-a2*b2*prev2 + (a2+b2)*k)) / (a2+b2));
        dm_table.push_back(dm);
    }
}

} // end namespace kernel
} // end namespace dedisp