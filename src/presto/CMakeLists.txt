add_library(libpresto SHARED
    barycenter_presto_utils.c
    barycenter.cpp
)

set_target_properties(libpresto PROPERTIES
    PUBLIC_HEADER barycenter_presto_utils.h 
    PUBLIC_HEADER barycenter.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

install(
    TARGETS
        libpresto
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)
