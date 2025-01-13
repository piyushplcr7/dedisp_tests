################################################################################
# test for dedisp c-library
################################################################################
add_executable(ctestdedisp
    ctestdedisp.c
)

target_include_directories(ctestdedisp PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(ctestdedisp dedisp) #dedisp c-library
target_link_libraries(ctestdedisp m) #c math library

##############################
# CLIG INTERFACE
##############################
add_library(fdd_gpu_clig fdd_gpu.c)


################################################################################
# test for dedisp library
################################################################################
add_executable(testdedisp
    testdedisp.cpp
)

target_include_directories(testdedisp PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(testdedisp fdd_gpu_clig dedisp ${CFITSIO_LIBRARY})

################################################################################
# test for read speed
################################################################################
add_executable(read_tests
    read_tests.cpp
)

target_include_directories(read_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(read_tests fdd_gpu_clig dedisp)

add_executable(fread_tests
    fread_tests.cpp
)

target_include_directories(fread_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(fread_tests fdd_gpu_clig dedisp)

add_executable(mmap_tests
    mmap_tests.cpp
)

target_include_directories(mmap_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(mmap_tests fdd_gpu_clig dedisp)

add_executable(th_read_test
    theoretical_read_test.cpp
)

target_include_directories(th_read_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(th_read_test fdd_gpu_clig dedisp)

add_executable(chunked_read
    chunked_read.cpp
)

target_include_directories(chunked_read PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(chunked_read fdd_gpu_clig dedisp)


################################################################################
# test for tdd library
################################################################################
add_executable(testtdd
    testtdd.cpp
)

target_include_directories(testtdd PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(testtdd tdd)


################################################################################
# test for fdd library
################################################################################
add_executable(testfdd
    testfdd.cpp
)

target_include_directories(testfdd PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(testfdd fdd)


################################################################################
# install
################################################################################
install(
    TARGETS
        ctestdedisp
        testdedisp
        testtdd
        testfdd
    RUNTIME
        DESTINATION bin
)