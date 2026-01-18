add_library(
    clib OBJECT
    dedisp.cpp
)

target_include_directories(
   clib
   PRIVATE ${CMAKE_SOURCE_DIR}/src
   PRIVATE ${GPU_INCLUDE_DIRS}
)

#target_link_libraries(
#    clib
#    ${GPU_FFT_LIBS}
#)