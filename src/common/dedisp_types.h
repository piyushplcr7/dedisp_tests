#ifndef DEDISP_TYPES_H_INCLUDE_GUARD
#define DEDISP_TYPES_H_INCLUDE_GUARD

// Types
// -----
typedef float                      dedisp_float;
typedef unsigned char              dedisp_byte;
typedef unsigned long              dedisp_size;
typedef int                        dedisp_bool;

// Internal word type used for transpose and dedispersion kernel
typedef unsigned int               dedisp_word;

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
/*! \typedef dedisp_plan
 * The plan type used by the library to reference a dedispersion plan.
     This is an opaque pointer type. */

// Summation type metafunction
template<int IN_NBITS> struct SumType { typedef dedisp_word type; };
// Note: For 32-bit input, we must accumulate using a larger data type
template<> struct SumType<32> { typedef unsigned long long type; };

#endif // DEDISP_TYPES_H_INCLUDE_GUARD