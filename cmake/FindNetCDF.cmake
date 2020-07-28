# - Find NetCDF
# Find the NetCDF C++ includes and library
#
#  NETCDF_INCLUDES    - where to find netcdf.h, etc
#  NETCDF_LIBRARIES   - Link these libraries when using NetCDF
#  NETCDF_FOUND       - True if NetCDF found including required interfaces (see below)
#
# Normal usage would be:
#  find_package (NetCDFcxx4 REQUIRED)
#  target_link_libraries (uses_f90_interface ${NETCDF_LIBRARIES})
#  target_link_libraries (only_uses_c_interface ${NETCDF_LIBRARIES_C})

find_path(NETCDF_F90_INCLUDES netcdf.mod
     HINTS ${netcdf_DIR}/include)
find_library(NETCDF_C_LIBRARY netcdf
     HINTS ${netcdf_DIR}/lib64 ${netcdf_DIR}/lib)
find_library(NETCDF_F90_LIBRARY netcdff
     HINTS ${netcdf_DIR}/lib64 ${netcdf_DIR}/lib)

set (NETCDF_LIBRARIES "${NetCDF_libs}" CACHE STRING "All NetCDF libraries required for interface level")
set (NETCDF_LIBRARIES ${NETCDF_C_LIBRARY} ${NETCDF_F90_LIBRARY})

# handle the QUIETLY and REQUIRED arguments and set NETCDF_FOUND to TRUE if
# all listed variables are TRUE
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (NetCDF DEFAULT_MSG NETCDF_F90_INCLUDES NETCDF_LIBRARIES)