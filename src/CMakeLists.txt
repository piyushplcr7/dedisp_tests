################################################################################
# helper libraries
################################################################################

# include directory for common header files
include_directories(common)

# add subdirectories for individual libraries
add_subdirectory(common)
add_subdirectory(external)


################################################################################
# dedisp library
################################################################################

add_subdirectory(dedisp)