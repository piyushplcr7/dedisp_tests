add_library(common OBJECT
    dedisp_error.cpp
)

set_property(TARGET common
    PROPERTY POSITION_INDEPENDENT_CODE on
)