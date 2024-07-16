# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/src/extern_re2"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/src/extern_re2-build"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/tmp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/src/extern_re2-stamp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/src"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/src/extern_re2-stamp"
)

set(configSubDirs Release)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/src/extern_re2-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/re2/src/extern_re2-stamp${cfgdir}") # cfgdir has leading slash
endif()
