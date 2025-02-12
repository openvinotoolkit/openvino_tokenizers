# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

include(FetchContent)

set(ICU_VERSION "70")
set(ICU_TARGET_NAME "icu_external")
set(ICU_URL https://github.com/unicode-org/icu/releases/download/release-70-1/icu4c-70_1-src.tgz)
set(ICU_URL_HASH SHA256=8d205428c17bf13bb535300669ed28b338a157b1c01ae66d31d0d3e2d47c3fd5)

set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/_deps/icu)
set(ICU_SOURCE_DIR  ${THIRD_PARTY_PATH}/icu-src)
set(ICU_INSTALL_DIR ${THIRD_PARTY_PATH}/icu-install)
set(ICU_BINARY_DIR ${THIRD_PARTY_PATH}/icu-target-build)

# required for cross-compilation
set(ICU_HOST_TARGET_NAME "icu_host_external")
set(ICU_HOST_INSTALL_DIR ${THIRD_PARTY_PATH}/icu-host-install)
set(ICU_HOST_BINARY_DIR ${THIRD_PARTY_PATH}/icu-host-build)

# ICU supports only Release and Debug build types
if(GENERATOR_IS_MULTI_CONFIG_VAR)
  list(APPEND ICU_CONFIGURE_FLAGS $<$<CONFIG:Debug>:"--enable-debug">$<$<NOT:$<CONFIG:Release>>:"--enable-release">)
  set(ICU_BUILD_TYPE $<$<CONFIG:Debug>:Debug>$<$<NOT:$<CONFIG:Debug>>:Release>)
else()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND ICU_CONFIGURE_FLAGS "--enable-debug")
    set(ICU_BUILD_TYPE ${CMAKE_BUILD_TYPE})
  else()
    list(APPEND ICU_CONFIGURE_FLAGS "--enable-release")
    set(ICU_BUILD_TYPE "Release")
  endif()
endif()

set(ICU_RELEASE_POSTFIX "")
if(WIN32)
  set(ICU_DEBUG_POSTFIX "d")
else()
  set(ICU_DEBUG_POSTFIX "")
endif()

# Define build artifacts

set(ICU_SHARED_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
set(ICU_STATIC_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
set(ICU_SHARED_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
set(ICU_STATIC_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})

if(WIN32)
    set(ICU_INSTALL_LIB_SUBDIR "lib64")
    set(ICU_INSTALL_BIN_SUBDIR "bin64")
    set(ICU_UC_LIB_NAME "icuuc")
    set(ICU_I18N_LIB_NAME "icuin")
    set(ICU_DATA_LIB_NAME "icudt")
    set(ICU_UC_SHARED_LIB_NAME "${ICU_UC_LIB_NAME}${ICU_VERSION}")
    set(ICU_I18N_SHARED_LIB_NAME "${ICU_I18N_LIB_NAME}${ICU_VERSION}")
    set(ICU_DATA_SHARED_LIB_NAME "${ICU_DATA_LIB_NAME}${ICU_VERSION}")
else()
    set(ICU_INSTALL_LIB_SUBDIR "lib")
    set(ICU_INSTALL_BIN_SUBDIR "lib")
    set(ICU_UC_LIB_NAME "icuuc")
    set(ICU_I18N_LIB_NAME "icui18n")
    set(ICU_DATA_LIB_NAME "icudata")
    set(ICU_UC_SHARED_LIB_NAME ${ICU_UC_LIB_NAME})
    set(ICU_I18N_SHARED_LIB_NAME ${ICU_I18N_LIB_NAME})
    set(ICU_DATA_SHARED_LIB_NAME ${ICU_DATA_LIB_NAME})

    # Calculate the number of cores using CMake
    execute_process(COMMAND nproc
      OUTPUT_VARIABLE ICU_JOB_POOL_SIZE
      OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

set(ICU_INCLUDE_DIRS "${ICU_INSTALL_DIR}/include")

# Compile flags

if(NOT WIN32)
  set(ICU_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-deprecated-declarations")
  set(ICU_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -Wno-deprecated-declarations")
endif()

# openvino::runtime exports _GLIBCXX_USE_CXX11_ABI=0 on CentOS7.
# It needs to be propagated to every library openvino_tokenizers links with.
# That prohibits linkage with prebuilt libraries because they aren't compiled with _GLIBCXX_USE_CXX11_ABI=0.
get_directory_property(OPENVINO_RUNTIME_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)

if(OPENVINO_RUNTIME_COMPILE_DEFINITIONS)
  foreach(def IN LISTS OPENVINO_RUNTIME_COMPILE_DEFINITIONS)
    set(ICU_CXX_FLAGS "${ICU_CXX_FLAGS} -D${def}")
    set(ICU_C_FLAGS "${ICU_C_FLAGS} -D${def}")
  endforeach()
endif()

# Build

#
# ov_tokenizer_build_icu(
#    TARGET_NAME <name>
#    TARGET_SYSTEM_NAME <name>
#    BUILD_ARCH <arch>
#    TARGET_ARCH <arch>
#    BUILD_DIR <dir>
#    INSTALL_DIR <dir>
#    [EXTRA_CONFIGURE_STEPS <step1 step2 ...>]
#    [HOST_ENV <env1 env2 ...>]
# )
#
function(ov_tokenizer_build_icu)
  set(oneValueRequiredArgs TARGET_NAME TARGET_SYSTEM_NAME BUILD_ARCH TARGET_ARCH BUILD_DIR INSTALL_DIR)
  set(multiValueArgs EXTRA_CONFIGURE_STEPS HOST_ENV)
  cmake_parse_arguments(ARG "" "${oneValueRequiredArgs};${oneValueOptionalArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_BUILD_ARCH STREQUAL ARG_TARGET_ARCH)
    set(ICU_CROSS_COMPILING ON)
  endif()

  # set target platform

  if(ARG_TARGET_SYSTEM_NAME MATCHES "^(Windows|WindowsCE|WindowsPhone|WindowsStore)$")
    set(ICU_WIN32 ON)
  elseif(ARG_TARGET_SYSTEM_NAME MATCHES "^(Darwin|iOS|tvOS|visionOS|watchOS)$")
    set(ICU_APPLE ON)
    set(ICU_UNIX ON)
  elseif(ARG_TARGET_SYSTEM_NAME STREQUAL "Android")
    set(ICU_ANDROID ON)
    set(ICU_UNIX ON)
  else()
    set(ICU_LINUX ON)
    set(ICU_UNIX ON)
  endif()

  foreach(build_type IN ITEMS Release Debug)
    string(TOUPPER ${build_type} BUILD_TYPE)

    unset(ICU_LIBRARIES_${BUILD_TYPE})
    unset(ICU_SHARED_LIBRARIES_${BUILD_TYPE})

    foreach(icu_target IN ITEMS UC I18N DATA)
      if(icu_target STREQUAL "DATA")
        set(lib_postfix ${ICU_RELEASE_POSTFIX})
      else()
        set(lib_postfix ${ICU_${BUILD_TYPE}_POSTFIX})
      endif()
      set(ICU_STATIC_LIB_DIR "${ARG_INSTALL_DIR}/${build_type}/${ICU_INSTALL_LIB_SUBDIR}")
      set(ICU_SHARED_LIB_DIR "${ARG_INSTALL_DIR}/${build_type}/${ICU_INSTALL_BIN_SUBDIR}")
      set(ICU_${icu_target}_LIB_${BUILD_TYPE} "${ICU_STATIC_LIB_DIR}/${ICU_STATIC_PREFIX}${ICU_${icu_target}_LIB_NAME}${lib_postfix}${ICU_STATIC_SUFFIX}")
      set(ICU_${icu_target}_SHARED_LIB_${BUILD_TYPE} "${ICU_SHARED_LIB_DIR}/${ICU_SHARED_PREFIX}${ICU_${icu_target}_SHARED_LIB_NAME}${lib_postfix}${ICU_SHARED_SUFFIX}")
      set(ICU_${icu_target}_LIB_${BUILD_TYPE} "${ICU_${icu_target}_LIB_${BUILD_TYPE}}" PARENT_SCOPE)
      set(ICU_${icu_target}_SHARED_LIB_${BUILD_TYPE} "${ICU_${icu_target}_SHARED_LIB_${BUILD_TYPE}}" PARENT_SCOPE)
      list(APPEND ICU_LIBRARIES_${BUILD_TYPE} ${ICU_${icu_target}_LIB_${BUILD_TYPE}})
      list(APPEND ICU_SHARED_LIBRARIES_${BUILD_TYPE} ${ICU_${icu_target}_SHARED_LIB_${BUILD_TYPE}})
    endforeach()

    set(ICU_LIBRARIES_${BUILD_TYPE} ${ICU_LIBRARIES_${BUILD_TYPE}} PARENT_SCOPE)
    set(ICU_SHARED_LIBRARIES_${BUILD_TYPE} ${ICU_SHARED_LIBRARIES_${BUILD_TYPE}} PARENT_SCOPE)
  endforeach()

  # need to unset several variables which can set env to cross-environment
  foreach(var SDKTARGETSYSROOT CONFIG_SITE OECORE_NATIVE_SYSROOT OECORE_TARGET_SYSROOT
              OECORE_ACLOCAL_OPTS OECORE_BASELIB OECORE_TARGET_ARCH OECORE_TARGET_OS CC CXX
              CPP AS LD GDB STRIP RANLIB OBJCOPY OBJDUMP READELF AR NM M4 TARGET_PREFIX
              CONFIGURE_FLAGS CFLAGS CXXFLAGS LDFLAGS CPPFLAGS KCFLAGS OECORE_DISTRO_VERSION
              OECORE_SDK_VERSION ARCH CROSS_COMPILE OE_CMAKE_TOOLCHAIN_FILE OPENSSL_CONF
              OE_CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_PATH)
    if(DEFINED ENV{${var}})
      set(ARG_HOST_ENV --unset=${var} ${ARG_HOST_ENV})
    endif()
  endforeach()

  set(ARG_HOST_ENV ${CMAKE_COMMAND} -E env ${ARG_HOST_ENV})

  include(ExternalProject)

  if(ICU_WIN32)
    ExternalProject_Add(
      ${ARG_TARGET_NAME}
      URL ${ICU_URL}
      URL_HASH ${ICU_URL_HASH}
      PREFIX ${THIRD_PARTY_PATH}
      SOURCE_DIR ${ICU_SOURCE_DIR}
      BINARY_DIR ${ARG_BUILD_DIR}
      INSTALL_DIR ${ARG_INSTALL_DIR}
      CONFIGURE_COMMAND ""
      BUILD_COMMAND msbuild ${ICU_SOURCE_DIR}\\source\\allinone\\allinone.sln /p:Configuration=${ICU_BUILD_TYPE} /p:Platform=x64 /t:i18n /t:uconv /t:makedata
      INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/include ${ARG_INSTALL_DIR}/include &&
                      ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/lib64 ${ARG_INSTALL_DIR}/${ICU_BUILD_TYPE}/${ICU_INSTALL_LIB_SUBDIR} &&
                      ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/bin64 ${ARG_INSTALL_DIR}/${ICU_BUILD_TYPE}/${ICU_INSTALL_BIN_SUBDIR}
      BUILD_BYPRODUCTS ${ICU_LIBRARIES_RELEASE} ${ICU_LIBRARIES_DEBUG})
  elseif(ICU_UNIX)
    set(ICU_CONFIGURE_FLAGS
      --prefix ${ARG_INSTALL_DIR}/${ICU_BUILD_TYPE}
      --includedir ${ICU_INCLUDE_DIRS}
      --enable-static
      --disable-rpath
      --disable-shared
      --disable-tests
      --disable-samples
      --disable-extras
      --disable-icuio
      --disable-draft
      --disable-icu-config
      --disable-layoutex
      --disable-layout
      ${ARG_EXTRA_CONFIGURE_STEPS})

    if(ICU_UNIX AND ICU_CROSS_COMPILING)
      if(ARG_TARGET_ARCH STREQUAL "X86_64")
        set(host_triplet x86_64-unknown-linux-gnu)
      elseif(ARG_TARGET_ARCH STREQUAL "AARCH64")
        set(host_triplet aarch64-unknown-linux-gnu)
      elseif(ARG_TARGET_ARCH STREQUAL "RISCV64")
        set(host_triplet riscv64-unknown-linux-gnu)
      else()
        message(FATAL_ERROR "Unsupported target arch: ${ARG_TARGET_ARCH}")
      endif()

      list(APPEND ICU_CONFIGURE_FLAGS --host=${host_triplet})
    endif()

    if(ICU_CROSS_COMPILING)
      list(APPEND ICU_CONFIGURE_FLAGS --disable-tools)
    else()
      list(APPEND ICU_CONFIGURE_FLAGS --enable-tools)
    endif()

    ExternalProject_Add(
      ${ARG_TARGET_NAME}
      URL ${ICU_URL}
      URL_HASH ${ICU_URL_HASH}
      PREFIX ${THIRD_PARTY_PATH}
      SOURCE_DIR ${ICU_SOURCE_DIR}
      BINARY_DIR ${ARG_BUILD_DIR}
      INSTALL_DIR ${ARG_INSTALL_DIR}
      CONFIGURE_COMMAND ${ARG_HOST_ENV} ${ICU_SOURCE_DIR}/source/configure ${ICU_CONFIGURE_FLAGS}
      BUILD_COMMAND make -j${ICU_JOB_POOL_SIZE}
      INSTALL_COMMAND make install
      BUILD_BYPRODUCTS ${ICU_LIBRARIES_RELEASE} ${ICU_LIBRARIES_DEBUG})
  else()
    message(FATAL_ERROR "Unsupported platform: ${ARG_TARGET_SYSTEM_NAME}")
  endif()
endfunction()

#
# Build
#

if(CMAKE_C_COMPILER_LAUNCHER)
  set(c_prefix "${CMAKE_C_COMPILER_LAUNCHER} ")
endif()

if(CMAKE_CXX_COMPILER_LAUNCHER)
  set(cxx_prefix "${CMAKE_CXX_COMPILER_LAUNCHER} ")
endif()

# Ensure native compilers are set
if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
  if(CMAKE_HOST_WIN32)
    find_program(CMAKE_C_COMPILER NAMES cl gcc clang)
    find_program(CMAKE_CXX_COMPILER NAMES cl g++ clang++)
  else()
    find_program(CMAKE_C_COMPILER NAMES cc gcc clang)
    find_program(CMAKE_CXX_COMPILER NAMES c++ g++ clang++)
  endif()

  if(NOT CMAKE_C_COMPILER OR NOT CMAKE_CXX_COMPILER)
    message(FATAL_ERROR "No C or C++ compiler found. Please install a compiler (GCC, Clang, or MSVC).")
  else()
    set(host_env_config
      CFLAGS=${c_prefix}${CMAKE_C_COMPILER}
      CXXFLAGS=${cxx_prefix}${CMAKE_CXX_COMPILER})
  endif()
endif()

# propogate current compilers and flags
if(APPLE)
  set(target_env_config
    CFLAGS=${ICU_C_FLAGS}
    CXXFLAGS=${ICU_CXX_FLAGS})
else()
  set(target_env_config
    CFLAGS=${ICU_C_FLAGS}
    CC=${cxx_prefix}${CMAKE_C_COMPILER}
    CXXFLAGS=${ICU_CXX_FLAGS}
    CXX=${cxx_prefix}${CMAKE_CXX_COMPILER})

    foreach(tool IN ITEMS CMAKE_AR CMAKE_RANLIB CMAKE_STRIP CMAKE_READELF CMAKE_OBJDUMP CMAKE_OBJCOPY
                      CMAKE_NM CMAKE_DLLTOOL CMAKE_ADDR2LINE CMAKE_MAKE_PROGRAM)
    set(tool ${tool})
    if(EXISTS ${tool})
      string(REPLACE "CMAKE_MAKE_PROGRAM" "MAKE" tool_name ${tool})
      string(REPLACE "CMAKE_LINKER" "LD" tool_name ${tool})
      string(REPLACE "CMAKE_" "" tool_name ${tool})
      list(APPEND target_env_config ${tool_name}=${tool})
    endif()
  endforeach()
endif()

if(NOT CMAKE_CROSSCOMPILING)
  set(host_env_config ${target_env_config})
  set(ICU_HOST_TARGET_NAME ${ICU_TARGET_NAME})
  set(ICU_HOST_INSTALL_DIR ${ICU_INSTALL_DIR})
  set(ICU_HOST_BINARY_DIR ${ICU_BINARY_DIR})
endif()

# build for host platform
ov_tokenizer_build_icu(
   TARGET_NAME ${ICU_HOST_TARGET_NAME}
   TARGET_SYSTEM_NAME ${CMAKE_HOST_SYSTEM_NAME}
   BUILD_ARCH ${OV_HOST_ARCH}
   TARGET_ARCH ${OV_HOST_ARCH}
   BUILD_DIR ${ICU_HOST_BINARY_DIR}
   INSTALL_DIR ${ICU_HOST_INSTALL_DIR}
   HOST_ENV ${host_env_config})

if(CMAKE_CROSSCOMPILING)
  # build for target platform
  ov_tokenizer_build_icu(
    TARGET_NAME ${ICU_TARGET_NAME}
    TARGET_SYSTEM_NAME ${CMAKE_SYSTEM_NAME}
    BUILD_ARCH ${OV_HOST_ARCH}
    TARGET_ARCH ${OV_ARCH}
    BUILD_DIR ${ICU_BINARY_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    EXTRA_CONFIGURE_STEPS --with-cross-build=${ICU_HOST_BINARY_DIR}
    HOST_ENV ${target_env_config})

    add_dependencies(${ICU_TARGET_NAME} ${ICU_HOST_TARGET_NAME})
endif()

# using custom FindICU module
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})
