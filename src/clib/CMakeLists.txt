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
    datacontainer
)

target_include_directories(clib 
    PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros
    PUBLIC ${CMAKE_SOURCE_DIR}/src/dataHandlers
)
target_compile_definitions(clib PRIVATE ${GPU_DEFINITIONS})