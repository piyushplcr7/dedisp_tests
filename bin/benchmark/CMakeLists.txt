################################################################################
# test for dedisp library
################################################################################
add_executable(benchdedisp
    benchdedisp.cpp
)

target_include_directories(benchdedisp PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(benchdedisp tdd)


################################################################################
# test for fdd library
################################################################################
add_executable(benchfdd
    benchfdd.cpp
)

target_include_directories(benchfdd PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(benchfdd fdd)


################################################################################
# install
################################################################################
install(
    TARGETS
        benchdedisp
        benchfdd
    RUNTIME
        DESTINATION bin
)