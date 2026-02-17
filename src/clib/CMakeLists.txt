add_library(
    clib OBJECT
    dedisp.cpp
)

target_include_directories(
   clib
   PRIVATE ${CMAKE_SOURCE_DIR}/src
   PRIVATE ${GPU_INCLUDE_DIRS}
   PRIVATE ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(clib PRIVATE 
    GPU_TRACER_LIBS
    ${CFITSIO_LIBRARY}
    GPU_XCCL
    )

target_include_directories(clib 
    PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros
    PUBLIC ${CMAKE_SOURCE_DIR}/src/fits
)
target_compile_definitions(clib PRIVATE ${GPU_DEFINITIONS})


#target_link_libraries(
#    clib
#    ${GPU_FFT_LIBS}
#)