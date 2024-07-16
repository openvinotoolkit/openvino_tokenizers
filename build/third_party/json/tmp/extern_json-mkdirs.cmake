# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/src/extern_json"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/src/extern_json-build"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/json"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/tmp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/src/extern_json-stamp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/src"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/src/extern_json-stamp"
)

set(configSubDirs Release)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/src/extern_json-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/json/src/extern_json-stamp${cfgdir}") # cfgdir has leading slash
endif()
