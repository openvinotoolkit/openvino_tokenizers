#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gflags::gflags_static" for configuration "Debug"
set_property(TARGET gflags::gflags_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gflags::gflags_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/gflags_static_debug.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS gflags::gflags_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_gflags::gflags_static "${_IMPORT_PREFIX}/lib/gflags_static_debug.lib" )

# Import target "gflags::gflags_nothreads_static" for configuration "Debug"
set_property(TARGET gflags::gflags_nothreads_static APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(gflags::gflags_nothreads_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/gflags_nothreads_static_debug.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS gflags::gflags_nothreads_static )
list(APPEND _IMPORT_CHECK_FILES_FOR_gflags::gflags_nothreads_static "${_IMPORT_PREFIX}/lib/gflags_nothreads_static_debug.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
