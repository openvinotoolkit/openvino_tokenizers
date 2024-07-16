# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/src/extern_gflags"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/src/extern_gflags-build"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/tmp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/src/extern_gflags-stamp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/src"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/src/extern_gflags-stamp"
)

set(configSubDirs Release)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/src/extern_gflags-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/gflags/src/extern_gflags-stamp${cfgdir}") # cfgdir has leading slash
endif()
