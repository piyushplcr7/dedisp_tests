# FindHWLOC.cmake
# Finds hwloc using HWLOC_ROOT or default paths

if(NOT DEFINED HWLOC_ROOT)
    set(HWLOC_ROOT "")
endif()

# --- Locate headers ---
if(HWLOC_ROOT)
    find_path(HWLOC_INCLUDE_DIR
        NAMES hwloc.h
        HINTS "${HWLOC_ROOT}/include"
    )
else()
    find_path(HWLOC_INCLUDE_DIR
        NAMES hwloc.h
    )
endif()

# --- Locate library ---
if(HWLOC_ROOT)
    find_library(HWLOC_LIBRARY
        NAMES hwloc
        HINTS "${HWLOC_ROOT}/lib"
    )
else()
    find_library(HWLOC_LIBRARY
        NAMES hwloc
    )
endif()

# --- Check if found ---
if(HWLOC_INCLUDE_DIR AND HWLOC_LIBRARY)
    set(HWLOC_FOUND TRUE)
else()
    set(HWLOC_FOUND FALSE)
endif()

if(NOT HWLOC_FOUND)
    message(FATAL_ERROR "Could not find hwloc. Please set HWLOC_ROOT or install hwloc.")
endif()

# --- Expose variables for target ---
set(HWLOC_LIBRARIES "${HWLOC_LIBRARY}")
set(HWLOC_INCLUDE_DIRS "${HWLOC_INCLUDE_DIR}")

add_library(hwloc::hwloc UNKNOWN IMPORTED)
set_target_properties(hwloc::hwloc PROPERTIES
    IMPORTED_LOCATION "${HWLOC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${HWLOC_INCLUDE_DIR}"
)
