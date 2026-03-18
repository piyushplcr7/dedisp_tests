#ifndef MPIUTILSHPP
#define MPIUTILSHPP

#define MPI_CHECK(call) do { \
  int _e = (call); \
  if (_e != MPI_SUCCESS) { \
    char es[512]; int el=0; MPI_Error_string(_e, es, &el); \
    fprintf(stderr, "MPI error %s:%d: %.*s\n", __FILE__, __LINE__, el, es); \
    MPI_Abort(MPI_COMM_WORLD, _e); \
  } \
} while (0)

template <typename T> MPI_Datatype mpi_type();

template <> inline MPI_Datatype mpi_type<float>()         { return MPI_FLOAT; }
template <> inline MPI_Datatype mpi_type<double>()        { return MPI_DOUBLE; }
template <> inline MPI_Datatype mpi_type<unsigned char>() { return MPI_UNSIGNED_CHAR; }
template <> inline MPI_Datatype mpi_type<int>()           { return MPI_INT; }


#endif