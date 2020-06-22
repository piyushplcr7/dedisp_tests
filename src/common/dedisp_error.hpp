#ifndef DEDISP_ERROR_H_INCLUDE_GUARD
#define DEDISP_ERROR_H_INCLUDE_GUARD

#include <sstream>
#include <stdexcept>

// Error codes
// -----------
typedef enum {
    DEDISP_NO_ERROR,
    DEDISP_MEM_ALLOC_FAILED,
    DEDISP_MEM_COPY_FAILED,
    DEDISP_NCHANS_EXCEEDS_LIMIT,
    DEDISP_INVALID_PLAN,
    DEDISP_INVALID_POINTER,
    DEDISP_INVALID_STRIDE,
    DEDISP_NO_DM_LIST_SET,
    DEDISP_TOO_FEW_NSAMPS,
    DEDISP_INVALID_FLAG_COMBINATION,
    DEDISP_UNSUPPORTED_IN_NBITS,
    DEDISP_UNSUPPORTED_OUT_NBITS,
    DEDISP_INVALID_DEVICE_INDEX,
    DEDISP_DEVICE_ALREADY_SET,
    DEDISP_PRIOR_GPU_ERROR,
    DEDISP_INTERNAL_GPU_ERROR,
    DEDISP_UNKNOWN_ERROR
} dedisp_error;

// Internal abstraction for errors

/*! \enum dedisp_error
 * Error codes for the library:\n
 * DEDISP_NO_ERROR: No error occurred.\n
 * DEDISP_MEM_ALLOC_FAILED: A memory allocation failed.\n
 * DEDISP_MEM_COPY_FAILED: A memory copy failed. This is often due to one of the arrays passed to dedisp_execute being too small.\n
 * DEDISP_NCHANS_EXCEEDS_LIMIT: The number of channels exceeds the internal limit. The current limit is 8192.\n
 * DEDISP_INVALID_PLAN: The given plan is NULL.\n
 * DEDISP_INVALID_POINTER: A pointer is invalid, possibly NULL.\n
 * DEDISP_INVALID_STRIDE: A stride value is less than the corresponding dimension's size.\n
 * DEDISP_NO_DM_LIST_SET: No DM list has yet been set using either \ref dedisp_set_dm_list or \ref dedisp_generate_dm_list.\n
 * DEDISP_TOO_FEW_NSAMPS: The number of time samples is less than the maximum dedispersion delay.\n
 * DEDISP_INVALID_FLAG_COMBINATION: Some of the given flags are incompatible.\n
 * DEDISP_UNSUPPORTED_IN_NBITS: The given \p in_nbits value is not supported. See \ref dedisp_execute for supported values.\n
 * DEDISP_UNSUPPORTED_OUT_NBITS: The given \p out_nbits value is not supported. See \ref dedisp_execute for supported values.\n
 * DEDISP_INVALID_DEVICE_INDEX: The given device index does not correspond to a device in the system.\n
 * DEDISP_DEVICE_ALREADY_SET: The device has already been set and cannot be changed. See \ref dedisp_set_device for more info.\n
 * DEDISP_PRIOR_GPU_ERROR: There was an existing GPU error prior to calling the function.\n
 * DEDISP_INTERNAL_GPU_ERROR: An unexpected GPU error has occurred within the library. Please contact the authors if you get this error.\n
 * DEDISP_UNKNOWN_ERROR: An unexpected error has occurred. Please contact the authors if you get this error.
 */

/*! \p dedisp_get_error_string gives a human-readable description of a
 *    given error code.
 *
 *  \param error The error code to describe.
 *  \return A string describing the error code.
 */
const char* dedisp_get_error_string(dedisp_error error);

#define throw_error(error) _throw_error(error, #error, __FILE__, __LINE__)

inline void _throw_error(
    dedisp_error error,
    char const *const func,
    char const *const file,
    int const line)
{
    if (error != DEDISP_NO_ERROR)
    {
        std::stringstream message;
        message << "An error occurred within dedisp on line ";
        message << line << " of " << file << ": ";
        message << dedisp_get_error_string(error);
        throw std::runtime_error(message.str());
    }
}

inline void check_error(
    dedisp_error error)
{
    throw_error(error);
}

#endif // DEDISP_ERROR_H_INCLUDE_GUARD