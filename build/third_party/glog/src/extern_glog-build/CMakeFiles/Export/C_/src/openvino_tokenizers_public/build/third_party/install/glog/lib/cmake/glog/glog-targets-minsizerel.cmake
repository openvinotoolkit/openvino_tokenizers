#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "glog::glog" for configuration "MinSizeRel"
set_property(TARGET glog::glog APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(glog::glog PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_MINSIZEREL "CXX"
  IMPORTED_LOCATION_MINSIZEREL "C:/src/openvino_tokenizers_public/build/third_party/install/glog/lib/glog.lib"
  )

list(APPEND _IMPORT_CHECK_TARGETS glog::glog )
list(APPEND _IMPORT_CHECK_FILES_FOR_glog::glog "C:/src/openvino_tokenizers_public/build/third_party/install/glog/lib/glog.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
