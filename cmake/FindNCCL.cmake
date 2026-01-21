# Find NCCL Library

find_path(
    NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ENV NCCL_ROOT 
    HINTS ENV NCCL_HOME 
    PATH_SUFFIXES include
)

find_library(
    NCCL_LIBRARY
    NAMES nccl
    HINTS ENV NCCL_ROOT
    HINTS ENV NCCL_HOME
    PATH_SUFFIXES lib lib/x86_64-linux-gnu
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  NCCL DEFAULT_MSG
  NCCL_LIBRARY
  NCCL_INCLUDE_DIR
)