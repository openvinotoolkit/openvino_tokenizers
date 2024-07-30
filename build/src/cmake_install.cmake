# Install script for directory: C:/src/openvino_tokenizers_public/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/openvino_tokenizers")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/cmake_install.cmake")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xopenvino_tokenizersx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/Debug/openvino_tokenizers.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/Release/openvino_tokenizers.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/MinSizeRel/openvino_tokenizers.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/RelWithDebInfo/openvino_tokenizers.dll")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xopenvino_tokenizersx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/fast_tokenizer/fast_tokenizer/Debug/core_tokenizers.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/fast_tokenizer/fast_tokenizer/Release/core_tokenizers.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/fast_tokenizer/fast_tokenizer/MinSizeRel/core_tokenizers.dll")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE SHARED_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/src/fast_tokenizer/fast_tokenizer/RelWithDebInfo/core_tokenizers.dll")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xopenvino_tokenizersx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/runtime/bin/intel64/Release" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/build/third_party/icu/src/extern_icu/icu4c/bin64/icudt70.dll"
    "C:/src/openvino_tokenizers_public/build/third_party/icu/src/extern_icu/icu4c/bin64/icuuc70.dll"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xopenvino_tokenizers_licensesx")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/-.dist-info" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/LICENSE"
    "C:/src/openvino_tokenizers_public/third-party-programs.txt"
    "C:/src/openvino_tokenizers_public/SECURITY.md"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xopenvino_tokenizers_docsx")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/docs/openvino_tokenizers" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/LICENSE"
    "C:/src/openvino_tokenizers_public/third-party-programs.txt"
    "C:/src/openvino_tokenizers_public/README.md"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xopenvino_tokenizers_pythonx")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/openvino_tokenizers" TYPE FILE FILES "C:/src/openvino_tokenizers_public/build/python/__version__.py")
endif()

