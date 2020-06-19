add_library(external OBJECT
    ${CMAKE_SOURCE_DIR}/src/external/Stopwatch.cpp
)

set_property(TARGET external
    PROPERTY POSITION_INDEPENDENT_CODE on)