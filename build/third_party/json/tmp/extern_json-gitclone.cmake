
if(NOT "C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json-stamp/extern_json-gitinfo.txt" IS_NEWER_THAN "C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json-stamp/extern_json-gitclone-lastrun.txt")
  message(STATUS "Avoiding repeated git clone, stamp file is up to date: 'C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json-stamp/extern_json-gitclone-lastrun.txt'")
  return()
endif()

execute_process(
  COMMAND ${CMAKE_COMMAND} -E rm -rf "C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to remove directory: 'C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json'")
endif()

# try the clone 3 times in case there is an odd git clone issue
set(error_code 1)
set(number_of_tries 0)
while(error_code AND number_of_tries LESS 3)
  execute_process(
    COMMAND "C:/Program Files/Git/cmd/git.exe"  clone --no-checkout --progress --config "advice.detachedHead=false" "https://github.com/nlohmann/json.git" "extern_json"
    WORKING_DIRECTORY "C:/src/openvino_tokenizers_public/build/third_party/json/src"
    RESULT_VARIABLE error_code
    )
  math(EXPR number_of_tries "${number_of_tries} + 1")
endwhile()
if(number_of_tries GREATER 1)
  message(STATUS "Had to git clone more than once:
          ${number_of_tries} times.")
endif()
if(error_code)
  message(FATAL_ERROR "Failed to clone repository: 'https://github.com/nlohmann/json.git'")
endif()

execute_process(
  COMMAND "C:/Program Files/Git/cmd/git.exe"  checkout v3.10.5 --
  WORKING_DIRECTORY "C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to checkout tag: 'v3.10.5'")
endif()

set(init_submodules TRUE)
if(init_submodules)
  execute_process(
    COMMAND "C:/Program Files/Git/cmd/git.exe"  submodule update --recursive --init 
    WORKING_DIRECTORY "C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json"
    RESULT_VARIABLE error_code
    )
endif()
if(error_code)
  message(FATAL_ERROR "Failed to update submodules in: 'C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json'")
endif()

# Complete success, update the script-last-run stamp file:
#
execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy
    "C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json-stamp/extern_json-gitinfo.txt"
    "C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json-stamp/extern_json-gitclone-lastrun.txt"
  RESULT_VARIABLE error_code
  )
if(error_code)
  message(FATAL_ERROR "Failed to copy script-last-run stamp file: 'C:/src/openvino_tokenizers_public/build/third_party/json/src/extern_json-stamp/extern_json-gitclone-lastrun.txt'")
endif()

