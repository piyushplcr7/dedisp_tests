#ifndef H_DEDISP_PLAN_INCLUDE_GUARD
#define H_DEDISP_PLAN_INCLUDE_GUARD

#include "GPUPlan.hpp"

namespace dedisp
{

class DedispPlan : public GPUPlan {

public:
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
               float_type df,
               int device_idx = 0);

    // No copying or assignment
    DedispPlan(const DedispPlan& other) = delete;
    DedispPlan& operator=(const DedispPlan& other) = delete;

    // Destructor
    /*! \p ~DedispPlan frees a plan and its associated resources
     *
     */
    ~DedispPlan();

    // Public interface
    void set_gulp_size(size_type gulp_size);
    size_type get_gulp_size() const { return m_gulp_size; };

    virtual void execute(size_type        nsamps,
                         const byte_type* in,
                         size_type        in_nbits,
                         byte_type*       out,
                         size_type        out_nbits)
    {
        execute(nsamps, in, in_nbits, out, out_nbits, 0);
    }

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

private:
    dedisp_size  m_gulp_size;
};

} // end namespace dedisp

#endif // H_DEDISP_PLAN_INCLUDE_GUARD