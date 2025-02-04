#ifndef DEDISP_TYPES_H_INCLUDE_GUARD
#define DEDISP_TYPES_H_INCLUDE_GUARD

// Types
// -----
typedef float                      dedisp_float;
typedef unsigned char              dedisp_byte;
typedef unsigned long              dedisp_size;
typedef int                        dedisp_bool;
/*! \typedef dedisp_float
 * The floating-point data-type used by the library. This is currently
     guaranteed to be equivalent to 'float'.*/
/*! \typedef dedisp_byte
 * The byte data-type used by the library to store time-series data. */
/*! \typedef dedisp_size
 * The size data-type used by the library to store sizes/dimensions. */
/*! \typedef dedisp_bool
 * The boolean data-type used by the library. Note that this type is
     implementation-defined and may not be equivalent to 'bool'. */

// Internal word type used for transpose and dedispersion kernel
typedef unsigned int               dedisp_word;

// Internal type used for complex numbers
typedef struct { float real; float imag; } dedisp_float2;

// Flags
// -----
typedef enum {
       DEDISP_USE_DEFAULT       = 0,
       DEDISP_HOST_POINTERS     = 1 << 1,
       DEDISP_DEVICE_POINTERS   = 1 << 2,

       DEDISP_WAIT              = 1 << 3,
       DEDISP_ASYNC             = 1 << 4
} dedisp_flag;
/*! \enum dedisp_flag
 * Flags for the library:\n
 * DEDISP_USE_DEFAULT: Use the default settings.\n
 * DEDISP_HOST_POINTERS: Instruct the function that the given pointers point to memory on the host.\n
 * DEDISP_DEVICE_POINTERS: Instruct the function that the given pointers point to memory on the device.\n
 * DEDISP_WAIT: Instruct the function to wait until all device operations are complete before returning.\n
 * DEDISP_ASYNC: Instruct the function to return before all device operations are complete.
 */

#endif // DEDISP_TYPES_H_INCLUDE_GUARD