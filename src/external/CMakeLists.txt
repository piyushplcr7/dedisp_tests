add_library(external OBJECT
    Stopwatch.cpp
)

set_property(TARGET external
    PROPERTY POSITION_INDEPENDENT_CODE on)