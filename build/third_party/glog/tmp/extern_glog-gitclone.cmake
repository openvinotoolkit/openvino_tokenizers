# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

if(EXISTS "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitclone-lastrun.txt" AND EXISTS "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitinfo.txt" AND
  "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitclone-lastrun.txt" IS_NEWER_THAN "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitinfo.txt")
  message(STATUS
    "Avoiding repeated git clone, stamp file is up to date: "
    "'/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitclone-lastrun.txt'"
  )
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: '/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "/usr/bin/git"
            clone --no-checkout --config "advice.detachedHead=false" "https://github.com/google/glog.git" "extern_glog"
    WORKING_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src"
    RESULT_VARIABLE error_code
  )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once: ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/google/glog.git'")
endif()

execute_process(
  COMMAND "/usr/bin/git"
          checkout "v0.4.0" --
  WORKING_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v0.4.0'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "/usr/bin/git" 
            submodule update --recursive --init 
    WORKING_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog"
    RESULT_VARIABLE error_code
  )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: '/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitinfo.txt" "/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
)
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: '/home/rmikhail/src/openvino_tokenizers/build/third_party/glog/src/extern_glog-stamp/extern_glog-gitclone-lastrun.txt'")
endif()
