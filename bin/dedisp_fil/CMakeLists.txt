add_executable(dedisp_fil
    dedisp_fil.cpp
)

target_include_directories(dedisp_fil PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/common
)

target_link_libraries(dedisp_fil dedisp)

install(
    TARGETS
        dedisp_fil
    RUNTIME
        DESTINATION bin
)