/*
  This is a simple C++ wrapper class for the dedisp library
*/

#pragma once

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>

#include <thrust/device_vector.h>

#include "dedisp_types.h"

namespace dedisp
{

class DedispPlan {

public:
    // Public types
    typedef dedisp_size  size_type;
    typedef dedisp_byte  byte_type;
    typedef dedisp_float float_type;
    typedef dedisp_bool  bool_type;

    // Constructor
    /*! \p DedispPlan builds a new plan object using the given parameters.
     *
     *  \param nchans Number of frequency channels
     *  \param dt Time difference between two consecutive samples in seconds
     *  \param f0 Frequency of the first (i.e., highest frequency) channel in MHz
     *  \param df Frequency difference between two consecutive channels in MHz
     *
     */
    DedispPlan(size_type  nchans,
               float_type dt,
               float_type f0,
               float_type df);

    // No copying or assignment
    DedispPlan(const DedispPlan& other) = delete;
    DedispPlan& operator=(const DedispPlan& other) = delete;

    // Destructor
    /*! \p ~DedispPlan frees a plan and its associated resources
     *
     */
    ~DedispPlan();

    static void set_device(int device_idx);

    // Public interface
    void set_gulp_size(size_type gulp_size);

    void set_killmask(const bool_type* killmask);

    void set_dm_list(const float_type* dm_list,
                     size_type         count);

    void generate_dm_list(float_type dm_start,
                          float_type dm_end,
                          float_type ti,
                          float_type tol);

    size_type         get_gulp_size()     const { return m_gulp_size; }
    float_type        get_max_delay()     const { return m_max_delay; }
    size_type         get_channel_count() const { return m_nchans; }
    size_type         get_dm_count()      const { return m_dm_count; }
    const float_type* get_dm_list()       const { return h_dm_list.data(); }
    const bool_type*  get_killmask()      const { return h_killmask.data(); }
    float_type        get_dt()            const { return m_dt; }
    float_type        get_df()            const { return m_df; }
    float_type        get_f0()            const { return m_f0; }


    void execute(size_type        nsamps,
                 const byte_type* in,
                 size_type        in_nbits,
                 byte_type*       out,
                 size_type        out_nbits,
                 unsigned         flags);

    void execute_adv(size_type        nsamps,
                     const byte_type* in,
                     size_type        in_nbits,
                     size_type        in_stride,
                     byte_type*       out,
                     size_type        out_nbits,
                     size_type        out_stride,
                     unsigned         flags);

    void execute_guru(size_type        nsamps,
                      const byte_type* in,
                      size_type        in_nbits,
                      size_type        in_stride,
                      byte_type*       out,
                      size_type        out_nbits,
                      size_type        out_stride,
                      dedisp_size      first_dm_idx,
                      dedisp_size      dm_count,
                      unsigned         flags);
    void sync();

private:
    // Size parameters
    dedisp_size  m_dm_count;
    dedisp_size  m_nchans;
    dedisp_size  m_max_delay;
    dedisp_size  m_gulp_size;

    // Physical parameters
    dedisp_float m_dt;
    dedisp_float m_f0;
    dedisp_float m_df;

    // Host arrays
    std::vector<dedisp_float> h_dm_list;      // size = dm_count
    std::vector<dedisp_float> h_delay_table;  // size = nchans
    std::vector<dedisp_bool>  h_killmask;     // size = nchans

    // Device arrays
    thrust::device_vector<dedisp_float> d_dm_list;
    thrust::device_vector<dedisp_float> d_delay_table;
    thrust::device_vector<dedisp_bool>  d_killmask;

    //StreamType m_stream;
};

} // end namespace dedisp