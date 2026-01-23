#ifndef MPICHECKMACRO
#define MPICHECKMACRO

#include <mpi.h>

#define MPICHECK(call)                                                       \
do {                                                                         \
    int _err = (call);                                                       \
    if (_err != MPI_SUCCESS) {                                               \
        char _errstr[MPI_MAX_ERROR_STRING];                                  \
        int _len;                                                            \
        MPI_Error_string(_err, _errstr, &_len);                              \
        fprintf(stderr, "MPI error at %s:%d - %s\n", __FILE__, __LINE__, _errstr); \
        MPI_Abort(MPI_COMM_WORLD, _err);                                     \
    }                                                                        \
} while (0)

#endif