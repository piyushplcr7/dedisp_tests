/*
* Generic GPUPlan class to be used with TDD and FDD GPU implementations.
* Generic GPU code from the original DedisPlan class was moved here.
* Code that is common to both CPU and GPU implementations is in the Plan class.
*/
#include "GPUPlan.hpp"

namespace dedisp
{

// Public interface
GPUPlan::GPUPlan(
    size_type  nchans,
    float_type dt,
    float_type f0,
    float_type df,
    int device_idx) :
    Plan(nchans, dt, f0, df)
{
    // Initialize device
    set_device(device_idx);

    // Initialize streams
    htodstream.reset(new cu::Stream());
    dtohstream.reset(new cu::Stream());
    executestream.reset(new cu::Stream());

    // Initialize delay table
    d_delay_table.resize(nchans * sizeof(dedisp_float));
    htodstream->memcpyHtoDAsync(d_delay_table, h_delay_table.data(), d_delay_table.size());

    // Initialize the killmask
    d_killmask.resize(nchans * sizeof(dedisp_bool));
    htodstream->memcpyHtoDAsync(d_killmask, h_killmask.data(), d_killmask.size());
}

// Destructor
GPUPlan::~GPUPlan()
{}

void GPUPlan::set_device(
    int device_idx)
{
    m_device.reset(new cu::Device(device_idx));
}

void GPUPlan::generate_dm_list(
    float_type dm_start,
    float_type dm_end,
    float_type ti,
    float_type tol)
{
    Plan::generate_dm_list(dm_start, dm_end, ti, tol);

    // Allocate device memory for the DM list
    d_dm_list.resize(m_dm_count * sizeof(dedisp_float));

    // Copy the DM list to the device
    htodstream->memcpyHtoDAsync(d_dm_list, h_dm_list.data(), d_dm_list.size());
}

void GPUPlan::set_dm_list(
    const float_type* dm_list,
    size_type         count)
{
    Plan::set_dm_list(dm_list, count);

    // Allocate device memory for the DM list
    d_dm_list.resize(m_dm_count * sizeof(dedisp_float));

    // Copy the DM list to the device
    htodstream->memcpyHtoDAsync(d_dm_list, h_dm_list.data(), d_dm_list.size());
}

void GPUPlan::set_killmask(
    const bool_type* killmask)
{
    Plan::set_killmask(killmask);

    // Copy the killmask to the device
    htodstream->memcpyHtoDAsync(d_killmask, h_killmask.data(), d_killmask.size());
}

} // end namespace dedisp