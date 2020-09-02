add_executable(testdedisp
    testdedisp.cpp
)

target_include_directories(testdedisp PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/tdd-ref
)

target_link_libraries(testdedisp dedisp)

install(
    TARGETS
        testdedisp
    RUNTIME
        DESTINATION bin
)