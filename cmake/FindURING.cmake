# Find liburing

find_path(
    URING_INCLUDE_DIR
    NAMES liburing.h
    HINTS ENV URING_ROOT
    HINTS ENV URING_HOME
    PATH_SUFFIXES include
)

find_library(
    URING_LIBRARY
    NAMES uring
    HINTS ENV URING_ROOT
    HINTS ENV URING_HOME
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  URING DEFAULT_MSG
  URING_LIBRARY
  URING_INCLUDE_DIR
)
