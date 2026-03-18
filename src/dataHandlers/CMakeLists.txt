add_library(datafile SHARED
    datafile.cpp
)

add_subdirectory(fits)
add_subdirectory(fil)

add_library(datacontainer SHARED
    fitscontainer.cpp
)


target_link_libraries(datacontainer
    MPI::MPI_CXX
    hwlocutils
    libpresto
    fits
    filterbank
)

target_include_directories(datacontainer
   PRIVATE ${MPI_CXX_INCLUDE_DIRS}
   PUBLIC ${CMAKE_SOURCE_DIR}/src/presto
   PUBLIC ${CMAKE_SOURCE_DIR}/src/numa
   PUBLIC ${CMAKE_SOURCE_DIR}/src/dataHandlers/fits
   PUBLIC ${CMAKE_SOURCE_DIR}/src/dataHandlers
)


set_target_properties(datacontainer PROPERTIES
    PUBLIC_HEADER fitscontainer.hpp
    VERSION ${DEDISP_VERSION}
    SOVERSION ${DEDISP_VERSION_MAJOR}
)

target_compile_features(datacontainer PRIVATE cxx_std_20)

install(
    TARGETS
        datacontainer
    LIBRARY
        DESTINATION lib
    PUBLIC_HEADER
        DESTINATION include
)
