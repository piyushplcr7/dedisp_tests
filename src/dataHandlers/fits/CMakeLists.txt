add_library(fits SHARED
    fits.cpp
)

target_link_libraries(fits
    PRIVATE ${CFITSIO_LIBRARY}
    PRIVATE libpresto
    PUBLIC datafile
)

target_include_directories(fits
   PRIVATE ${CFITSIO_INCLUDE_DIR}
   PRIVATE ${CMAKE_SOURCE_DIR}/src/presto
   PUBLIC ${CMAKE_SOURCE_DIR}/src/dataHandlers
)

set_target_properties(fits PROPERTIES
    PUBLIC_HEADER fits.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

target_compile_features(fits PRIVATE cxx_std_20)

install(
    TARGETS
        fits
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)
