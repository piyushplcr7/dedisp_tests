# Find CFITSIO Library

find_path(
    CFITSIO_INCLUDE_DIR
    NAMES fitsio.h
    HINTS ENV CFITSIO_ROOT_DIR 
    HINTS ${CFITSIO_ROOT_DIR} 
    PATH_SUFFIXES include
)

find_library(
    CFITSIO_LIBRARY
    NAMES cfitsio
    HINTS ENV CFITSIO_ROOT_DIR
    HINTS ${CFITSIO_ROOT_DIR}
    PATH_SUFFIXES lib lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  CFITSIO_INCLUDE_DIR DEFAULT_MSG
  CFITSIO_INCLUDE_DIR
)

find_package_handle_standard_args(
  CFITSIO_LIBRARY DEFAULT_MSG
  CFITSIO_LIBRARY
)