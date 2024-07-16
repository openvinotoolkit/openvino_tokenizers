# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src/extern_gflags")
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src/extern_gflags")
endif()
file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src/extern_gflags-build"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/tmp"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src/extern_gflags-stamp"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src/extern_gflags-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src/extern_gflags-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/third_party/gflags/src/extern_gflags-stamp${cfgdir}") # cfgdir has leading slash
endif()
