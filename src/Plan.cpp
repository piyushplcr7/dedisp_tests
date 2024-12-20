/*
* Generic Plan class to be used with TDD and FDD implementations
* Generic code from the original DedisPlan class was moved here,
* with the intention that alternative dedispersion implementations
* can be added without too much duplication.
*/
#include "Plan.hpp"
#include <cmath>
#include "common/dedisp_error.hpp"
#include "common/cuda/CU.h"

namespace dedisp
{

// Public interface
Plan::Plan(
    size_type  nchans,
    float_type dt,
    float_type f0,
    float_type df)
{
    m_dm_count      = 0;
    m_nchans        = nchans;
    m_max_delay     = 0;
    m_dt            = dt;
    m_f0            = f0;
    m_df            = df;

    // Force the df parameter to be negative such that
    //   freq[chan] = f0 + chan * df.
    df = -std::abs(df);

    // Generate delay table and copy to device memory
    // Note: The DM factor is left out and applied during dedispersion
    h_delay_table.resize(nchans);
    generate_delay_table(h_delay_table.data(), nchans, dt, f0, df);

    // Initialize the killmask
    h_killmask.resize(nchans, (dedisp_bool)true);
}

Plan::~Plan()
{}

void Plan::generate_dm_list(
    std::vector<dedisp_float>& dm_table,
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
        double dm = ((b2*prev + std::sqrt(-a2*b2*prev2 + (a2+b2)*k)) / (a2+b2));
        dm_table.push_back(dm);
    }
}

void Plan::set_dm_list(
    const float_type* dm_list,
    size_type         count)
{
    if (!dm_list)
    {
        throw_error(DEDISP_INVALID_POINTER);
    }

    m_dm_count = count;
    h_dm_list.assign(dm_list, dm_list+count);

    // Calculate the maximum delay
    m_max_delay = dedisp_size(h_dm_list[m_dm_count-1] *
                              h_delay_table[m_nchans-1] + 0.5);
}

void Plan::set_killmask(
    const bool_type* killmask)
{
    if (0 != killmask)
    {
        // Set killmask
        h_killmask.assign(killmask, killmask + m_nchans);
    } else
    {
        // Set the killmask to all true
        std::fill(h_killmask.begin(), h_killmask.end(), (dedisp_bool)true);
    }
}

void Plan::sync()
{
    cu::checkError(cudaDeviceSynchronize());
}

// Private helper functions
void Plan::generate_delay_table(
    dedisp_float* h_delay_table, dedisp_size nchans,
    dedisp_float dt, dedisp_float f0, dedisp_float df)
{
    for( dedisp_size c=0; c<nchans; ++c ) {
        dedisp_float a = 1.f / (f0+c*df);
        dedisp_float b = 1.f / f0;
        // Note: To higher precision, the constant is 4.148741601e3
        h_delay_table[c] = 4.15e3/dt * (a*a - b*b);
    }
}

dedisp_float Plan::get_smearing(
    dedisp_float dt, dedisp_float pulse_width,
    dedisp_float f0, dedisp_size nchans, dedisp_float df,
    dedisp_float DM, dedisp_float deltaDM)
{
    dedisp_float W         = pulse_width;
    dedisp_float BW        = nchans * abs(df);
    dedisp_float fc        = f0 - BW/2;
    dedisp_float inv_fc3   = 1./(fc*fc*fc);
    dedisp_float t_DM      = 8.3*BW*DM*inv_fc3;
    dedisp_float t_deltaDM = 8.3/4*BW*nchans*deltaDM*inv_fc3;
    dedisp_float t_smear   = std::sqrt(dt*dt + W*W + t_DM*t_DM + t_deltaDM*t_deltaDM);
    return t_smear;
}

void Plan::generate_dm_list(
    float_type dm_start,
    float_type dm_end,
    float_type ti,
    float_type tol)
{
    // Generate the DM list
    h_dm_list.clear();
    generate_dm_list(
        h_dm_list,
        dm_start, dm_end,
        m_dt, ti, m_f0, m_df,
        m_nchans, tol);
    m_dm_count = h_dm_list.size();

    // Calculate the maximum delay and store it in the plan
    m_max_delay = dedisp_size(h_dm_list[m_dm_count-1] *
                              h_delay_table[m_nchans-1] + 0.5);
}

} // end namespace dedisp
