# - Try to find the CUDA nvToolsExt library
# Variables used by this module:
#  CUDA_TOOLKIT_ROOT_DIR   - CUDA toolkit root directory
# Variables defined by this module:
#  CUDA_nvToolsExt_FOUND   - system has CUDA_nvToolsExt_LIBRARY
#  CUDA_nvToolsExt_LIBRARY - the CUDA_nvToolsExt_LIBRARY library

if(NOT CUDA_nvToolsExt_LIBRARY_FOUND)

find_library(
  CUDA_nvToolsExt_LIBRARY
  NAMES nvToolsExt
  HINTS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDA_nvToolsExt_LIBRARY DEFAULT_MSG CUDA_nvToolsExt_LIBRARY)

endif(NOT CUDA_nvToolsExt_LIBRARY_FOUND)