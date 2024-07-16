#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "glog::glog" for configuration ""
set_property(TARGET glog::glog APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(glog::glog PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/install/glog/lib/libglog.a"
  )

list(APPEND _cmake_import_check_targets glog::glog )
list(APPEND _cmake_import_check_files_for_glog::glog "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/install/glog/lib/libglog.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
