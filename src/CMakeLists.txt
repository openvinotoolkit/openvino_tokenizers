# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "openvino_tokenizers")

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0126)
  cmake_policy(SET CMP0126 NEW)
endif()

if(POLICY CMP0127)
  cmake_policy(SET CMP0127 NEW)
endif()

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

if(POLICY CMP0169)
  cmake_policy(SET CMP0169 OLD)
endif()

function(ov_tokenizers_set_flags flags replace_value replace_pattern)
  foreach(flag ${flags})
    if(${flag} MATCHES "${replace_pattern}")
      string(REGEX REPLACE "${replace_pattern}" "${replace_value}" ${flag} "${${flag}}")
    elseif(NOT (${flag} MATCHES "${replace_value}"))
      set(${flag} "${${flag}} ${replace_value}")
    endif()
    set(${flag} "${${flag}}" PARENT_SCOPE)
  endforeach()
endfunction()

function(ov_tokenizers_set_cxx_standard)
ov_tokenizers_set_flags("CMAKE_CXX_FLAGS;CMAKE_CXX_FLAGS_DEBUG;CMAKE_CXX_FLAGS_RELEASE;CMAKE_CXX_FLAGS_MINSIZEREL;CMAKE_CXX_FLAGS_RELWITHDEBINFO" "" "--std=c\\+\\+11")
  set(CMAKE_CXX_STANDARD 17 PARENT_SCOPE)
  set(CMAKE_CXX_EXTENSIONS OFF PARENT_SCOPE)
  set(CMAKE_CXX_STANDARD_REQUIRED ON PARENT_SCOPE)
endfunction()

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0")
endif()

ov_tokenizers_set_cxx_standard()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# openvino::runtime exports _GLIBCXX_USE_CXX11_ABI=0 on CentOS7.
# It needs to be propagated to every library openvino_tokenizers links with.
# That prohibits linkage with prebuilt libraries because they aren't compiled with _GLIBCXX_USE_CXX11_ABI=0.
get_directory_property(OPENVINO_RUNTIME_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
include(CMakeDependentOption)

#
# Compile flags
#

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "^(Apple)?Clang$")
    set(c_cxx_flags "-Wno-undef")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # C4146: unary minus operator applied to unsigned type, result still unsigned
    # C4244: 'argument' : conversion from 'type1' to 'type2', possible loss of data
    # C4267: 'var' : conversion from 'size_t' to 'type', possible loss of data
    # C4700: uninitialized local variable 'var' used
    set(c_cxx_flags "/wd4146 /wd4244 /wd4267 /wd4700 /wd4703")
endif()

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-Wsuggest-override" SUGGEST_OVERRIDE_SUPPORTED)
if(SUGGEST_OVERRIDE_SUPPORTED)
    set(cxx_flags "${cxx_flags} -Wno-suggest-override")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cxx_flags} ${c_cxx_flags}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${c_cxx_flags}")

# Apply for 'Release' explicitly for WA '/sdl' issues
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${cxx_flags} ${c_cxx_flags}")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${c_cxx_flags}")

if(WIN32 OR APPLE)
  set(CMAKE_DEBUG_POSTFIX "d")
endif()

#
# Dependencies
#

include(FetchContent)

option(ENABLE_SYSTEM_ICU "Enables use of system ICU" OFF)
set(SUPPORTED_ARCHS "X86;X86_64")
# There are no prebuilt ICU packages for macOS and Linux arm64
if (ENABLE_SYSTEM_ICU OR APPLE OR NOT OV_HOST_ARCH IN_LIST SUPPORTED_ARCHS) 
  message(STATUS "Using system-installed ICU.")
else()
  if(UNIX)
    FetchContent_Declare(
      ICU
      URL https://github.com/unicode-org/icu/releases/download/release-70-1/icu4c-70_1-Ubuntu-20.04-x64.tgz
      URL_HASH SHA256=a8134e9f8a68d33600749601e143e553b5cb48c217c8941dbb9ef478fac420dd
    )
  elseif(WIN32)
    FetchContent_Declare(
      ICU
      URL https://github.com/unicode-org/icu/releases/download/release-70-1/icu4c-70_1-Win64-MSVC2019.zip
      URL_HASH SHA256=af6b585e49d90d39ae9d3fe298b7f56983931706a5e49d4bce675c6a499124e5
    )
  endif()
  FetchContent_MakeAvailable(ICU)
  set(ICU_ROOT "${CMAKE_BINARY_DIR}/_deps/icu-src/usr/local")
  message(STATUS "Using prebuilt ICU from ${ICU_ROOT}.")
endif()

FetchContent_Declare(
  sentencepiece
  URL      https://github.com/google/sentencepiece/archive/d8f741853847553169444afc12c00f4bbff3e9ce.tar.gz
  URL_HASH SHA256=1cf6e0713ecd04d1dd3328fdd388aa89c8ebab518a15e0886b54eadd8d228886
)
FetchContent_GetProperties(sentencepiece)
if(NOT sentencepiece_POPULATED)
if(DEFINED ENV{CONDA_BUILD_SYSROOT})
    set(openvino_installed_from_conda ON)
    # OpenVINO conda package dynamically linked with external protobuf,
    # and we need to link sentencepiece with external protobuf too.
    set(CMAKE_FIND_PACKAGE_PREFER_CONFIG ON)
    set(protobuf_MODULE_COMPATIBLE ON CACHE BOOL "Protobuf module compatible")
endif()
if(openvino_installed_from_conda AND NOT WIN32)
    set(SPM_USE_BUILTIN_PROTOBUF OFF CACHE BOOL "")
    set(SPM_PROTOBUF_PROVIDER "package" CACHE STRING "")
    set(SPM_ABSL_PROVIDER "package" CACHE STRING "")
else()
    set(SPM_USE_BUILTIN_PROTOBUF ON CACHE BOOL "")
    set(SPM_PROTOBUF_PROVIDER "internal" CACHE STRING "")
    set(SPM_ABSL_PROVIDER "internal" CACHE STRING "")
endif()

set(SPM_ENABLE_SHARED OFF CACHE BOOL "")
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "")
set(SPM_ENABLE_NFKC_COMPILE ON CACHE BOOL "Enable NFKC compile")
FetchContent_Populate(sentencepiece)
add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

function(ov_tokenizers_link_sentencepiece TARGET_NAME)
  if(sentencepiece_FOUND)
    foreach(sp_target sentencepiece sentencepiece_train)
      if(TARGET ${sp_target}-static)
        # on Windows conda-forge builds sentencepiece as static library
        target_link_libraries(${TARGET_NAME} PRIVATE ${sp_target}-static)
      else()
        target_link_libraries(${TARGET_NAME} PRIVATE ${sp_target})
      endif()
    endforeach()
    target_link_libraries(${TARGET_NAME} PRIVATE absl::string_view absl::flat_hash_set)
  else()
    target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
      "${sentencepiece_SOURCE_DIR}/src/builtin_pb"
      "${sentencepiece_SOURCE_DIR}/src"
      "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite"
      "${sentencepiece_SOURCE_DIR}/third_party/" # for libabseil
      "${sentencepiece_SOURCE_DIR}"
      "${sentencepiece_BINARY_DIR}")

    foreach(sp_target sentencepiece-static sentencepiece_train-static)
      if(CMAKE_CL_64)
        target_compile_definitions(${sp_target} PRIVATE _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
      endif()
      # to propogate _GLIBCXX_USE_CXX11_ABI value
      target_compile_definitions(${sp_target} PUBLIC $<TARGET_PROPERTY:openvino::runtime,INTERFACE_COMPILE_DEFINITIONS>)
      target_link_libraries(${TARGET_NAME} PRIVATE ${sp_target})
    endforeach(sp_target sentencepiece sentencepiece_train)

    if(ANDROID)
      # see https://github.com/protocolbuffers/protobuf/issues/2719#issuecomment-625400968
      target_link_libraries(${TARGET_NAME} PRIVATE log)
    endif()
  endif()
endfunction()

function(ov_tokenizers_build_static_re2)
  FetchContent_Declare(
    re2
    URL      https://github.com/google/re2/archive/refs/tags/2022-04-01.tar.gz
    URL_HASH SHA256=1ae8ccfdb1066a731bba6ee0881baad5efd2cd661acd9569b689f2586e1a50e9
  )
  set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
  set(BUILD_SHARED_LIBS OFF)
  set(RE2_BUILD_TESTING OFF)
  FetchContent_GetProperties(re2)
  if(NOT re2_POPULATED)
    FetchContent_Populate(re2)
    add_subdirectory(${re2_SOURCE_DIR} ${re2_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  # to propogate _GLIBCXX_USE_CXX11_ABI value
  target_compile_definitions(re2 PUBLIC $<TARGET_PROPERTY:openvino::runtime,INTERFACE_COMPILE_DEFINITIONS>)
endfunction()

ov_tokenizers_build_static_re2()

function(ov_tokenizers_link_pcre2 TARGET_NAME)
  FetchContent_Declare(
      prce2
      URL https://github.com/PCRE2Project/pcre2/archive/refs/tags/pcre2-10.44.zip
      URL_HASH SHA256=2d87bd1700bd1993ddea7c56aad2b0373ac2b3d52d9cc78842a6d061ffaf0925
  )
  FetchContent_GetProperties(prce2)
  if(NOT prce2_POPULATED)
    FetchContent_Populate(prce2)

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(PCRE2_STATIC_PIC ON)
    set(PCRE2_BUILD_TESTS OFF)
    set(PCRE2_BUILD_PCRE2GREP OFF)
    set(PCRE2_BUILD_TESTS)

    add_subdirectory(${prce2_SOURCE_DIR} ${prce2_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${pcre2_BINARY_DIR})
  target_link_libraries(${TARGET_NAME} PRIVATE pcre2-8)
  target_compile_definitions(${TARGET_NAME} PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
endfunction()

function(ov_tokenizers_link_re2 TARGET_NAME)
    target_link_libraries(${TARGET_NAME} PRIVATE re2)
endfunction()

#
# Build library
#

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_library(${TARGET_NAME} SHARED ${SRCS})

#
# Target include dirs, link libraries and other properties
#

ov_tokenizers_link_sentencepiece(${TARGET_NAME})
ov_tokenizers_link_pcre2(${TARGET_NAME})
ov_tokenizers_link_re2(${TARGET_NAME})

string(REPLACE " " ";" extra_flags "${c_cxx_flags} ${cxx_flags}")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_OPTIONS "${extra_flags}")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime openvino::threading)

#
# Set install RPATH
#

# setting RPATH / LC_RPATH depending on platform
if(LINUX)
  # to find libcore_tokenizer.so in the same folder
  set(rpaths "$ORIGIN")
elseif(APPLE)
  # to find libcore_tokenizer.dylib in the same folder
  set(rpaths "@loader_path")
  if(DEFINED PY_BUILD_CMAKE_PACKAGE_NAME)
    # in case we build pip package, we need to refer to libopenvino.dylib from 'openvino' package
    list(APPEND rpaths "@loader_path/../../openvino/libs")
  endif()
endif()

if(rpaths)
  set_target_properties(${TARGET_NAME} PROPERTIES INSTALL_RPATH "${rpaths}")
endif()

#
# Installation
#

# Put binaries at the top level for NPM package
if(CPACK_GENERATOR STREQUAL "NPM")
  set(OPENVINO_TOKENIZERS_INSTALL_LIBDIR .)
  set(OPENVINO_TOKENIZERS_INSTALL_BINDIR .)
else()
  # - Windows: `<openvino_dir>\runtime\bin\intel64\Release\`
  # - MacOS_x86: `<openvino_dir>/runtime/lib/intel64/Release`
  # - MacOS_arm64: `<openvino_dir>/runtime/lib/arm64/Release/`
  # - Linux_x86: `<openvino_dir>/runtime/lib/intel64/`
  # - Linux_arm64: `<openvino_dir>/runtime/lib/aarch64/`
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" OPENVINO_TOKENIZERS_INSTALL_DIR)
  if(OPENVINO_TOKENIZERS_INSTALL_DIR MATCHES "amd64.*|x86_64.*|AMD64.*")
    set(OPENVINO_TOKENIZERS_INSTALL_DIR intel64)
  elseif(OPENVINO_TOKENIZERS_INSTALL_DIR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
    if(APPLE)
      set(OPENVINO_TOKENIZERS_INSTALL_DIR "arm64")
    else()
      set(OPENVINO_TOKENIZERS_INSTALL_DIR "aarch64")
    endif()
  elseif(OPENVINO_TOKENIZERS_INSTALL_DIR STREQUAL "x86_64" OR OPENVINO_TOKENIZERS_INSTALL_DIR STREQUAL "amd64"  # Windows detects Intel's 64-bit CPU as AMD64
      OR CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
    set(OPENVINO_TOKENIZERS_INSTALL_DIR intel64)
  endif()

  if(WIN32 OR APPLE)
    set(OPENVINO_TOKENIZERS_INSTALL_DIR ${OPENVINO_TOKENIZERS_INSTALL_DIR}/${BUILD_TYPE})
  endif()

  set(OPENVINO_TOKENIZERS_INSTALL_BINDIR "runtime/bin/${OPENVINO_TOKENIZERS_INSTALL_DIR}" CACHE STRING "Destination for files installation of bin files - Windows dll")
  set(OPENVINO_TOKENIZERS_INSTALL_LIBDIR "runtime/lib/${OPENVINO_TOKENIZERS_INSTALL_DIR}" CACHE STRING  "Destination for files installation of lib files")
endif()

# Installing the extension module
install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_LIBDIR} COMPONENT openvino_tokenizers
        RUNTIME DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_BINDIR} COMPONENT openvino_tokenizers)

if(extra_libs)
  if(WIN32)
    set(extra_libs_location ${OPENVINO_TOKENIZERS_INSTALL_BINDIR})
  else()
    set(extra_libs_location ${OPENVINO_TOKENIZERS_INSTALL_LIBDIR})
  endif()
  install(FILES ${extra_libs} DESTINATION ${extra_libs_location} COMPONENT openvino_tokenizers)
endif()

install(FILES "${openvino_tokenizers_SOURCE_DIR}/LICENSE"
              "${openvino_tokenizers_SOURCE_DIR}/third-party-programs.txt"
              "${openvino_tokenizers_SOURCE_DIR}/README.md"
        DESTINATION "docs/openvino_tokenizers"
        COMPONENT openvino_tokenizers_docs)

#
# Cpack configuration
#

set(CPACK_COMPONENTS_ALL openvino_tokenizers openvino_tokenizers_docs)

# Uniform outcome in all repos - all repos will not create top level directory
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)

include (CPack)
