################################################################################
# dedisp_fil binary
################################################################################
add_executable(dedisp_fil
    dedisp_fil.cpp
)

target_include_directories(dedisp_fil PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(dedisp_fil dedisp)


################################################################################
# tdd_fil binary
################################################################################
add_executable(tdd_fil
    tdd_fil.cpp
)

target_include_directories(tdd_fil PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(tdd_fil tdd)


################################################################################
# fdd_fil binary
################################################################################
add_executable(fdd_fil
    fdd_fil.cpp
)

target_include_directories(fdd_fil PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

target_link_libraries(fdd_fil fdd)


################################################################################
# install
################################################################################
install(
    TARGETS
        dedisp_fil
        tdd_fil
        fdd_fil
    RUNTIME
        DESTINATION bin
)