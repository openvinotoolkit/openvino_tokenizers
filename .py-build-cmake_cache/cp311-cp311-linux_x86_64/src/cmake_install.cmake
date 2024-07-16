# Install script for directory: /home/rmikhail/src/openvino_tokenizers/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/tmp/tmp0yg28idr/staging")
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

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set path to fallback-tool for dependency-resolution.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_tokenizers" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libopenvino_tokenizers.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libopenvino_tokenizers.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libopenvino_tokenizers.so"
         RPATH "$ORIGIN")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib" TYPE SHARED_LIBRARY FILES "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/src/libopenvino_tokenizers.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libopenvino_tokenizers.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libopenvino_tokenizers.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libopenvino_tokenizers.so"
         OLD_RPATH "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/src/fast_tokenizer/fast_tokenizer:/tmp/pip-build-env-4yaqcir1/overlay/lib/python3.11/site-packages/openvino/libs:"
         NEW_RPATH "$ORIGIN")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libopenvino_tokenizers.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_tokenizers" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_tokenizers" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libcore_tokenizers.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libcore_tokenizers.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libcore_tokenizers.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib" TYPE SHARED_LIBRARY FILES "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/src/fast_tokenizer/fast_tokenizer/libcore_tokenizers.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libcore_tokenizers.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libcore_tokenizers.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/openvino_tokenizers/lib/libcore_tokenizers.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_tokenizers" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_tokenizers_licenses")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/openvino_tokenizers-2024.3.0.0.dist-info" TYPE FILE FILES
    "/home/rmikhail/src/openvino_tokenizers/LICENSE"
    "/home/rmikhail/src/openvino_tokenizers/third-party-programs.txt"
    "/home/rmikhail/src/openvino_tokenizers/SECURITY.md"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "openvino_tokenizers_python")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/openvino_tokenizers" TYPE FILE FILES "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/python/__version__.py")
endif()

