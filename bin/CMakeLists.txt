# make sure that the subsequent binaries can find the dedisp library
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# add binaries
add_subdirectory(test)
add_subdirectory(dedisp_fil)