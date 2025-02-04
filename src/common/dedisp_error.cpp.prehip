#include "dedisp_error.hpp"

const char* dedisp_get_error_string(dedisp_error error)
{
    switch( error ) {
    case DEDISP_NO_ERROR:
        return "No error";
    case DEDISP_MEM_ALLOC_FAILED:
        return "Memory allocation failed";
    case DEDISP_MEM_COPY_FAILED:
        return "Memory copy failed";
    case DEDISP_INVALID_DEVICE_INDEX:
        return "Invalid device index";
    case DEDISP_DEVICE_ALREADY_SET:
        return "Device is already set and cannot be changed";
    case DEDISP_NCHANS_EXCEEDS_LIMIT:
        return "No. channels exceeds internal limit";
    case DEDISP_INVALID_PLAN:
        return "Invalid plan";
    case DEDISP_INVALID_POINTER:
        return "Invalid pointer";
    case DEDISP_INVALID_STRIDE:
        return "Invalid stride";
    case DEDISP_NO_DM_LIST_SET:
        return "No DM list has been set";
    case DEDISP_TOO_FEW_NSAMPS:
        return "No. samples < maximum delay";
    case DEDISP_INVALID_FLAG_COMBINATION:
        return "Invalid flag combination";
    case DEDISP_UNSUPPORTED_IN_NBITS:
        return "Unsupported in_nbits value";
    case DEDISP_UNSUPPORTED_OUT_NBITS:
        return "Unsupported out_nbits value";
    case DEDISP_PRIOR_GPU_ERROR:
        return "Prior GPU error.";
    case DEDISP_INTERNAL_GPU_ERROR:
        return "Internal GPU error. Please contact the author(s).";
    case DEDISP_UNKNOWN_ERROR:
        return "Unknown error. Please contact the author(s).";
    default:
        return "Invalid error code";
    }
}