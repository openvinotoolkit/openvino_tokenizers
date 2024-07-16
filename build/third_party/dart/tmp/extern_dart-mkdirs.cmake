# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/src/extern_dart"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/src/extern_dart-build"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/tmp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/src/extern_dart-stamp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/src"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/src/extern_dart-stamp"
)

set(configSubDirs Release)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/src/extern_dart-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/dart/src/extern_dart-stamp${cfgdir}") # cfgdir has leading slash
endif()
