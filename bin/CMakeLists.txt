# make sure that the subsequent binaries can find the dedisp library
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# add binaries
add_subdirectory(test)

if(NOT GPU_BACKEND STREQUAL "OPENMP")
add_subdirectory(benchmark)
add_subdirectory(fil)
endif()
