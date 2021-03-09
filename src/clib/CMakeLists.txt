add_library(clib OBJECT
    dedisp.c
)

target_link_libraries(
    clib
)

target_include_directories(
   clib
   PRIVATE ${CMAKE_SOURCE_DIR}/src
)

set_target_properties(
    clib PROPERTIES PUBLIC_HEADER dedisp.h
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

install(
    TARGETS
        clib
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)