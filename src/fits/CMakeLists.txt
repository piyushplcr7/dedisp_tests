add_library(fits SHARED
    fits.cpp
    fitscontainer.cpp
)

##############################
# CLIG INTERFACE
##############################

target_link_libraries(fits
    ${CFITSIO_LIBRARY}
    MPI::MPI_CXX
)

target_include_directories(fits
   PRIVATE ${CFITSIO_INCLUDE_DIR}
   PRIVATE ${MPI_CXX_INCLUDE_DIRS}
)

set_target_properties(fits PROPERTIES
    PUBLIC_HEADER fits.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

install(
    TARGETS
        fits
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)
