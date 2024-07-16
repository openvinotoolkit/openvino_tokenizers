# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-build"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/tmp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src"
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp"
)

set(configSubDirs Release)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp${cfgdir}") # cfgdir has leading slash
endif()
