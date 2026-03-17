add_library(filterbank SHARED
    fil.cpp
)

target_link_libraries(filterbank
    libpresto)

set_target_properties(filterbank PROPERTIES
    PUBLIC_HEADER fil.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

target_compile_features(fits PRIVATE cxx_std_20)

install(
    TARGETS
        filterbank
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)
