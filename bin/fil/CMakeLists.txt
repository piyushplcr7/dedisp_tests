################################################################################
# dedisp_fil binary
################################################################################
add_executable(dedisp_fil
    dedisp_fil.cpp
)

target_include_directories(dedisp_fil PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_include_directories(dedisp_fil PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(dedisp_fil PRIVATE ${GPU_DEFINITIONS})

target_link_libraries(dedisp_fil PRIVATE dedisp)
target_link_libraries(dedisp_fil PRIVATE GPU_TRACER_LIBS)


################################################################################
# tdd_fil binary
################################################################################
add_executable(tdd_fil
    tdd_fil.cpp
)

target_include_directories(tdd_fil PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_include_directories(tdd_fil PUBLIC ${CMAKE_SOURCE_DIR}/src/gpu_macros)
target_compile_definitions(tdd_fil PRIVATE ${GPU_DEFINITIONS})

target_link_libraries(tdd_fil PRIVATE tdd)
target_link_libraries(tdd_fil PRIVATE GPU_TRACER_LIBS)


################################################################################
# fdd_fil binary
################################################################################
#add_executable(fdd_fil
#    fdd_fil.cpp
#)

#target_include_directories(fdd_fil PRIVATE
#    ${CMAKE_SOURCE_DIR}/src
#)

#target_link_libraries(fdd_fil fdd)

################################################################################
# fil_class_test binary
################################################################################
add_executable(fil_class_test
    fil_class_test.cpp
)

target_include_directories(fil_class_test PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(fil_class_test filterbank datafile)

################################################################################
# reduce_fil binary (debug only)
################################################################################
if(TESTDEDISP_DEBUG)
add_executable(reduce_fil
    reduce_fil.cpp
)

target_include_directories(reduce_fil PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(reduce_fil filterbank datafile)
endif()

################################################################################
# install
################################################################################
set(_FIL_INSTALL_TARGETS
    dedisp_fil
    tdd_fil
    fil_class_test
)
if(TESTDEDISP_DEBUG)
    list(APPEND _FIL_INSTALL_TARGETS reduce_fil)
endif()

install(
    TARGETS
        ${_FIL_INSTALL_TARGETS}
    RUNTIME
        DESTINATION bin
)