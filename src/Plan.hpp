/*
* Generic Plan class to be used with TDD and FDD implementations
* Generic code from the original DedisPlan class was moved here,
* with the intention that alternative dedispersion implementations
* can be added without too much duplication.
*/
#ifndef DEDISP_PLAN_H_INCLUDE_GUARD
#define DEDISP_PLAN_H_INCLUDE_GUARD

#include <vector>
#include <memory>

#include "common/dedisp_types.h"
#include "common/cuda/CU.h"

namespace dedisp
{

class Plan {

public:
    // Public types
    typedef dedisp_size  size_type;
    typedef dedisp_byte  byte_type;
    typedef dedisp_float float_type;
    typedef dedisp_bool  bool_type;

    // Constructor
    /*! \p Plan builds a new plan object using the given parameters.
     *
     *  \param nchans Number of frequency channels
     *  \param dt Time difference between two consecutive samples in seconds
     *  \param f0 Frequency of the first (i.e., highest frequency) channel in MHz
     *  \param df Frequency difference between two consecutive channels in MHz
     *
     */
    Plan(
        size_type  nchans,
        float_type dt,
        float_type f0,
        float_type df);

    // No copying or assignment
    Plan(const Plan& other) = delete;
    Plan& operator=(const Plan& other) = delete;

    // Destructor
    virtual ~Plan();

    // Public interface (common)
    void generate_dm_list(
        float_type dm_start,
        float_type dm_end,
        float_type ti,
        float_type tol);

    void set_dm_list(
        const float_type* dm_list,
        size_type count);

    void set_killmask(
        const bool_type* killmask);

    float_type        get_max_delay()     const { return m_max_delay; }
    size_type         get_channel_count() const { return m_nchans; }
    size_type         get_dm_count()      const { return m_dm_count; }
    float_type        get_dt()            const { return m_dt; }
    float_type        get_df()            const { return m_df; }
    float_type        get_f0()            const { return m_f0; }

    const float_type* get_dm_list()  const { return h_dm_list.data(); }
    const bool_type*  get_killmask() const { return h_killmask.data(); }

    void sync();

    // Public interface (virtual)
    virtual void execute(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits) = 0;

    virtual void set_device(int device_idx = 0) {};

private:
    // Helper methods
    void generate_delay_table(dedisp_float* h_delay_table, dedisp_size nchans,
                              dedisp_float dt, dedisp_float f0, dedisp_float df);

    dedisp_float get_smearing(dedisp_float dt, dedisp_float pulse_width,
                              dedisp_float f0, dedisp_size nchans, dedisp_float df,
                              dedisp_float DM, dedisp_float deltaDM);

    void generate_dm_list(std::vector<dedisp_float>& dm_table,
                          dedisp_float dm_start, dedisp_float dm_end,
                          double dt, double ti, double f0, double df,
                          dedisp_size nchans, double tol);

protected:
    // Size parameters
    dedisp_size  m_dm_count;
    dedisp_size  m_nchans;
    dedisp_size  m_max_delay;

    // Physical parameters
    dedisp_float m_dt;
    dedisp_float m_f0;
    dedisp_float m_df;

    // Host arrays
    std::vector<dedisp_float> h_dm_list;      // size = dm_count
    std::vector<dedisp_float> h_delay_table;  // size = nchans
    std::vector<dedisp_bool>  h_killmask;     // size = nchans
};

} // end namespace dedisp

#endif // DEDISP_PLAN_H_INCLUDE_GUARD