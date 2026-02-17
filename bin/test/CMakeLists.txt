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

include_directories(${CMAKE_SOURCE_DIR}/src/fdd)

################################################################################
# test for barycenter using libpresto
################################################################################
add_executable(barycenter_test
    barycenter_test.cpp
)

target_include_directories(barycenter_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/presto
)

target_link_libraries(barycenter_test libpresto) #dedisp c-library

target_compile_features(barycenter_test PRIVATE cxx_std_20)

#target_compile_options(barycenter_test PRIVATE)

add_executable(resample_test
    resample_map_test.cpp
)

target_include_directories(resample_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src/presto
)

target_link_libraries(resample_test libpresto) #dedisp c-library

target_compile_features(resample_test PRIVATE cxx_std_20)

################################################################################
# test for dedisp library
################################################################################
add_executable(testdedisp
    testdedisp_new.cpp
)

target_include_directories(testdedisp PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_include_directories(testdedisp PUBLIC ${CMAKE_SOURCE_DIR}/src/numa)
target_include_directories(testdedisp PUBLIC ${CMAKE_SOURCE_DIR}/src/common)

target_compile_definitions(testdedisp PRIVATE ${GPU_DEFINITIONS})

target_link_libraries(testdedisp PRIVATE GPU_TRACER_LIBS)
target_link_libraries(testdedisp PRIVATE GPU_XCCL)
target_link_libraries(testdedisp PRIVATE hwlocutils)
target_link_libraries(testdedisp PRIVATE ${GPU_HOST_RUNTIME_LIBS})

target_include_directories(testdedisp 
    PRIVATE ${CMAKE_SOURCE_DIR}/src 
    PRIVATE ${CFITSIO_INCLUDE_DIR} ${CMAKE_SOURCE_DIR}/src/presto
)

target_link_libraries(testdedisp PRIVATE
    fdd_gpu_clig 
    fdd 
    libpresto
    ${CFITSIO_LIBRARY} 
    ${GPU_FFT_LIBS}
    MPI::MPI_CXX
    fits
)

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
    ${CMAKE_SOURCE_DIR}/src ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(chunked_read fdd_gpu_clig dedisp ${CFITSIO_LIBRARY})

add_executable(mmap_only_read
    mmap_only_read.cpp
)

target_include_directories(mmap_only_read PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(mmap_only_read fdd_gpu_clig dedisp ${CFITSIO_LIBRARY})

add_executable(fits_class_test
    fits_class_test.cpp
)

target_include_directories(fits_class_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(fits_class_test fits ${CFITSIO_LIBRARY})

add_executable(fits_class_test_mpi
    fits_class_test_mpi.cpp
)

target_include_directories(fits_class_test_mpi PRIVATE
    ${CMAKE_SOURCE_DIR}/src 
    ${CFITSIO_INCLUDE_DIR}
    ${MPI_CXX_INCLUDE_DIRS}
)

target_link_libraries(fits_class_test_mpi fits ${CFITSIO_LIBRARY} MPI::MPI_CXX)

add_executable(numatests
    numatests.cpp
)

target_include_directories(numatests PRIVATE
    ${CMAKE_SOURCE_DIR}/src 
    ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(numatests fits ${CFITSIO_LIBRARY})

################################################################################
set_source_files_properties(
    numa_gpu_bw.cu
    PROPERTIES LANGUAGE ${GPU_LANG}
)

add_executable(numa_gpu_bw
    numa_gpu_bw.cu
)

target_include_directories(numa_gpu_bw PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(numa_gpu_bw PRIVATE ${GPU_DEFINITIONS})

target_include_directories(numa_gpu_bw PRIVATE
    ${CMAKE_SOURCE_DIR}/src 
)

target_link_libraries(numa_gpu_bw hwlocutils)

################################################################################
set_source_files_properties(
    numa_mul_gpu_bw.cu
    PROPERTIES LANGUAGE ${GPU_LANG}
)

add_executable(numa_mul_gpu_bw
    numa_mul_gpu_bw.cu
)

target_include_directories(numa_mul_gpu_bw PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(numa_mul_gpu_bw PRIVATE ${GPU_DEFINITIONS})

target_include_directories(numa_mul_gpu_bw PRIVATE
    ${CMAKE_SOURCE_DIR}/src 
    ${CMAKE_SOURCE_DIR}/src/gpu_macros
)

target_link_libraries(numa_mul_gpu_bw hwlocutils)

################################################################################

add_executable(numa_bw
    numa_bw.cpp
)

target_link_libraries(numa_bw numa)

################################################################################

add_executable(parFitsLoad
    parFitsLoad.cpp
)

target_include_directories(parFitsLoad PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CFITSIO_INCLUDE_DIR}
)

target_link_libraries(parFitsLoad fits hwloc::hwloc numa)

################################################################################
# test for tdd library
################################################################################
add_executable(testtdd
    testtdd.cpp
)

target_include_directories(testtdd PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(testtdd PRIVATE ${GPU_DEFINITIONS})

target_include_directories(testtdd PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(testtdd PRIVATE tdd)
target_link_libraries(testtdd PRIVATE GPU_TRACER_LIBS)


################################################################################
# test for fdd library
################################################################################
#add_executable(testfdd
#    testfdd.cpp
#)

#target_include_directories(testfdd PRIVATE
#    ${CMAKE_SOURCE_DIR}/src ${NCCL_INCLUDE_DIR}
#)

#target_link_libraries(testfdd fdd ${NCCL_LIBRARY})


################################################################################
# install
################################################################################
install(
    TARGETS
        ctestdedisp
        #testdedisp
        testtdd
        #testfdd
    RUNTIME
        DESTINATION bin
)