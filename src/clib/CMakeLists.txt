add_library(
    clib OBJECT
    dedisp.cpp
)

target_include_directories(
   clib
   PRIVATE ${CMAKE_SOURCE_DIR}/src
)