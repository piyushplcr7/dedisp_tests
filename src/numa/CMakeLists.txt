add_library(hwlocutils SHARED
    hwloc_utils.cpp
    numa_mem.cpp
)

target_link_libraries(hwlocutils
    numa
    hwloc::hwloc
)

set_target_properties(hwlocutils PROPERTIES
    PUBLIC_HEADER hwloc_utils.hpp
    PUBLIC_HEADER numa_mem.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

install(
    TARGETS
        hwlocutils
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)
