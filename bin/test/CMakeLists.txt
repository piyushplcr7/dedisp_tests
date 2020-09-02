################################################################################
# test for dedisp library
################################################################################
add_executable(testdedisp
    testdedisp.cpp
)

target_include_directories(testdedisp PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(testdedisp dedisp)


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
        testdedisp
        testtdd
        testfdd
    RUNTIME
        DESTINATION bin
)