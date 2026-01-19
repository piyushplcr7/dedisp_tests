add_library(
    clib OBJECT
    dedisp.cpp
)

target_include_directories(
   clib
   PRIVATE ${CMAKE_SOURCE_DIR}/src
   PRIVATE ${GPU_INCLUDE_DIRS}
)

target_include_directories(clib PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(clib PRIVATE ${GPU_DEFINITIONS})


#target_link_libraries(
#    clib
#    ${GPU_FFT_LIBS}
#)