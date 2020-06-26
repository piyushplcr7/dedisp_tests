# - Try to find the FFTW3 library
# Variables used by this module:
#  FFTW3_ROOT_DIR   - CUDA toolkit root directory
# Variables defined by this module:
#  FFTW3_FOUND   - system has FFTW3_LIBRARY
#  FFTW3_LIBRARY - the FFTW3_LIBRARY library
#  FFTW3F_FOUND   - system has FFTW3F_LIBRARY
#  FFTW3F_LIBRARY - the FFTW3F_LIBRARY library

find_path(
  FFTW3_INCLUDE_DIR
  NAMES fftw3.h
  HINTS ENV FFTW3_ROOT_DIR PATH_SUFFIXES include
  HINTS ${FFTW3_ROOT_DIR} PATH_SUFFIXES include
)

find_library(
  FFTW3_LIBRARY
  NAMES fftw3
  HINTS ENV FFTW3_ROOT_DIR
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib
)

find_library(
  FFTW3F_LIBRARY
  NAMES fftw3f
  HINTS ENV FFTW3_ROOT_DIR
  HINTS ${FFTW3_ROOT_DIR}
  PATH_SUFFIXES lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  FFTW3_INCLUDE_DIR DEFAULT_MSG
  FFTW3_INCLUDE_DIR
)

find_package_handle_standard_args(
  FFTW3_LIBRARY DEFAULT_MSG
  FFTW3_LIBRARY
)

find_package_handle_standard_args(
  FFTW3F_LIBRARY DEFAULT_MSG
  FFTW3F_LIBRARY
)