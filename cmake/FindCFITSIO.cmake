find_package(PkgConfig)
pkg_check_modules(PC_CFITSIO QUIET cfitsio)

#if (PC_CFITSIO_FOUND)
#    set(CFITSIO_INCLUDE_DIR ${PC_CFITSIO_INCLUDE_DIRS})
#    set(CFITSIO_LIBRARY ${PC_CFITSIO_LIBRARIES})
#endif()

message(STATUS "CFITSIO_INCLUDE_DIR after pkgconfig: ${CFITSIO_INCLUDE_DIR}")
message(STATUS "CFITSIO_LIBRARY after pkgconfig: ${CFITSIO_LIBRARY}")

find_path(
    CFITSIO_INCLUDE_DIR
    NAMES fitsio.h
    HINTS ENV CFITSIO_ROOT_DIR ${CFITSIO_ROOT_DIR}
          ${PC_CFITSIO_INCLUDE_DIRS}
    PATH_SUFFIXES include
)

find_library(
    CFITSIO_LIBRARY
    NAMES cfitsio
    HINTS ENV CFITSIO_ROOT_DIR ${CFITSIO_ROOT_DIR}
          ${PC_CFITSIO_LIBRARY_DIRS}
    PATH_SUFFIXES lib lib/x86_64-linux-gnu
)

message(STATUS "CFITSIO_INCLUDE_DIR after find_path: ${CFITSIO_INCLUDE_DIR}")
message(STATUS "CFITSIO_LIBRARY after find_library: ${CFITSIO_LIBRARY}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CFITSIO
  REQUIRED_VARS CFITSIO_LIBRARY CFITSIO_INCLUDE_DIR
  FAIL_MESSAGE "Could not find CFITSIO library or headers"
)
