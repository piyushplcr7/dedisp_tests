/*
* Generic GPUPlan class to be used with TDD and FDD GPU implementations.
* Generic GPU code from the original DedisPlan class was moved here.
* Code that is common to both CPU and GPU implementations is in the Plan class.
*/
#ifndef DEDISP_PLAN_H_GPU_INCLUDE_GUARD
#define DEDISP_PLAN_H_GPU_INCLUDE_GUARD

#include "Plan.hpp"

namespace dedisp
{

class GPUPlan : public Plan
{
protected:
    // Constructor
    /*! \p Plan builds a new plan object using the given parameters.
     *
     *  \param nchans Number of frequency channels
     *  \param dt Time difference between two consecutive samples in seconds
     *  \param f0 Frequency of the first (i.e., highest frequency) channel in MHz
     *  \param df Frequency difference between two consecutive channels in MHz
     *  \param device_idx Select which GPU to use, default = 0
     *
     */
    GPUPlan(
        size_type  nchans,
        float_type dt,
        float_type f0,
        float_type df,
        int device_idx = 0);

    // No copying or assignment
    GPUPlan(const GPUPlan& other) = delete;
    GPUPlan& operator=(const GPUPlan& other) = delete;

    // Destructor
    virtual ~GPUPlan();

    // Device
    void set_device(int device_idx);
    std::unique_ptr<cu::Device> m_device;

    // Device arrays
    cu::DeviceMemory d_dm_list;     // type = dedisp_float
    cu::DeviceMemory d_delay_table; // type = dedisp_float
    cu::DeviceMemory d_killmask;    // type = dedisp_bool

    // Streams
    std::unique_ptr<cu::Stream> htodstream;
    std::unique_ptr<cu::Stream> dtohstream;
    std::unique_ptr<cu::Stream> executestream;

public:
    // Public interface
    virtual void generate_dm_list(
        float_type dm_start,
        float_type dm_end,
        float_type ti,
        float_type tol) override;

    virtual void set_dm_list(
        const float_type* dm_list,
        size_type count) override;

    virtual void set_killmask(
        const bool_type* killmask) override;

};

} // end namespace dedisp

#endif // DEDISP_PLAN_GPU_H_INCLUDE_GUARD