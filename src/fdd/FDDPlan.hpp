#include "GPUPlan.hpp"

namespace dedisp
{

class FDDPlan : public GPUPlan {

public:
    // Constructor
    FDDPlan(
        size_type  nchans,
        float_type dt,
        float_type f0,
        float_type df);

    // Destructor
    ~FDDPlan();

    // Public interface
    virtual void execute(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

private:
    // Private interface
    virtual void execute_cpu(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    virtual void execute_cpu_segmented(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    virtual void execute_gpu(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    virtual void execute_gpu_segmented(
        size_type        nsamps,
        const byte_type* in,
        size_type        in_nbits,
        byte_type*       out,
        size_type        out_nbits);

    // Helper methods
    void generate_spin_frequency_table(
        dedisp_size nfreq,
        dedisp_size nsamp,
        dedisp_float dt);

    // Host arrays
    std::vector<dedisp_float> h_spin_frequencies; // size = nfreq

    // Device arrays
    cu::DeviceMemory d_spin_frequencies; // type = dedisp_float
};

} // end namespace dedisp