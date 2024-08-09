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

function(ov_tokenizers_set_cxx_standard)
  foreach(build_type "" "_DEBUG" "_MINSIZEREL" "_RELEASE" "_RELWITHDEBINFO")
    set(flag_var "CMAKE_CXX_FLAGS${build_type}")
    string(REPLACE "--std=c++11" "" ${flag_var} "${${flag_var}}")
    set(${flag_var} "${${flag_var}}" PARENT_SCOPE)
  endforeach()

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
if("_GLIBCXX_USE_CXX11_ABI=0" IN_LIST OPENVINO_RUNTIME_COMPILE_DEFINITIONS)
  set(USE_ABI0 ON CACHE BOOL "Set -D_GLIBCXX_USE_CXX11_ABI to 0 for fast_tokenizers")
endif()

if(ANDROID)
  if(AARCH64 OR ARM)
    set(FAST_TOKENIZERS_SUPPORTED ON)
    if(ANDROID_NATIVE_API_LEVEL LESS 33)
      message(FATAL_ERROR "FastTokenizers require ANDROID_NATIVE_API_LEVEL to be higher than 33. Please, either disable FastTokenizers or set ANDROID_NATIVE_API_LEVEL / ANDROID_PLATFORM")
    endif()
  elseif(X86_64)
    message(WARNING "FastTokenizers are not available on ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}. ENABLE_FAST_TOKENIZERS is set to OFF")
    set(FAST_TOKENIZERS_SUPPORTED OFF)
  else()
    message(WARNING "Unsupport Android ${CMAKE_SYSTEM_PROCESSOR}. Please, contact OpenVINO Tokenizers developers")
  endif()
  set(FAST_TOKENIZER_FROM_SOURCES ON)
else()
  set(FAST_TOKENIZERS_SUPPORTED ON)
  if(USE_ABI0)
    set(FAST_TOKENIZER_FROM_SOURCES ON)
  else()
    set(FAST_TOKENIZER_FROM_SOURCES OFF)
  endif()
endif()

cmake_dependent_option(ENABLE_FAST_TOKENIZERS "Enables Fast Tokenizers usage in OpenVINO Tokenizers" ON "FAST_TOKENIZERS_SUPPORTED" OFF)

if(ENABLE_FAST_TOKENIZERS)
  # The option is forced to ON if _GLIBCXX_USE_CXX11_ABI=0 or on Android (where prebuilt version is not available)
  cmake_dependent_option(BUILD_FAST_TOKENIZERS "Compile core_tokenizers instead of downloading prebuilt library" OFF "NOT FAST_TOKENIZER_FROM_SOURCES" ON)
  if(NOT BUILD_FAST_TOKENIZERS AND NOT CMAKE_BUILD_TYPE MATCHES Release)
    message(FATAL_ERROR "The ${CMAKE_BUILD_TYPE} build could not be completed without rebuilding the fast tokenizers. Please manually set BUILD_FAST_TOKENIZERS=ON." )
  endif()
else()
  set(BUILD_FAST_TOKENIZERS OFF CACHE BOOL "Compile core_tokenizers instead of downloading prebuilt library" FORCE)
endif()

#
# Compile flags
#

# Make sure to not link debug libs
# against release libs and vice versa
if(WIN32 AND CMAKE_BUILD_TYPE MATCHES Debug)
  set(CMAKE_DEBUG_POSTFIX "d")
endif()

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

if(BUILD_FAST_TOKENIZERS)
  # FastTokenizers use static MSVC runtime, hence we have to use it as well
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

  if(WIN32)
    foreach(
        flag_var
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_C_FLAGS_DEBUG)
        if(${flag_var} MATCHES "/MD")
          string(REGEX REPLACE "/MD" "/MTd" ${flag_var} "${${flag_var}}")
        elseif(NOT (${flag_var} MATCHES "/MTd"))
          set(${flag_var} "${${flag_var}} /MTd")
        endif()
        set(${flag_var} "${${flag_var}}" CACHE STRING "msvc compiler flags" FORCE)
    endforeach()
  endif()
endif()

#
# Dependencies
#

include(FetchContent)

find_package(sentencepiece QUIET)
if(sentencepiece_FOUND)
  find_package(absl REQUIRED)
else()
  FetchContent_Declare(
    sentencepiece
    URL      https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz
    URL_HASH SHA256=9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86
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
    FetchContent_Populate(sentencepiece)
    add_subdirectory(${sentencepiece_SOURCE_DIR} ${sentencepiece_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

function(ov_tokenizers_link_sentencepiece TARGET_NAME)
  if(sentencepiece_FOUND)
    if(TARGET sentencepiece-static)
      # on Windows conda-forge builds sentencepiece as static library
      set(sp_target sentencepiece-static)
    else()
      set(sp_target sentencepiece)
    endif()
    target_link_libraries(${TARGET_NAME} PRIVATE ${sp_target} absl::string_view absl::flat_hash_set)
  else()
    target_include_directories(${TARGET_NAME} PRIVATE
      "${sentencepiece_SOURCE_DIR}/src/builtin_pb"
      "${sentencepiece_SOURCE_DIR}/src"
      "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite"
      "${sentencepiece_SOURCE_DIR}/third_party/" # for libabseil
      "${sentencepiece_SOURCE_DIR}"
      "${sentencepiece_BINARY_DIR}")

    if(CMAKE_CL_64)
        target_compile_definitions(sentencepiece-static PRIVATE _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
    endif()
    target_compile_definitions(sentencepiece-static PUBLIC $<TARGET_PROPERTY:openvino::runtime,INTERFACE_COMPILE_DEFINITIONS>)
    target_link_libraries(${TARGET_NAME} PRIVATE sentencepiece-static)
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
endfunction()

if(BUILD_FAST_TOKENIZERS)
  set(EXTERNAL_PROJECT_SOURCE_DIR ${CMAKE_BINARY_DIR}/_deps/fast_tokenizer/src)
  set(EXTERNAL_PROJECT_BINARY_DIR ${CMAKE_BINARY_DIR}/_deps/fast_tokenizer/build)
  set(EXTERNAL_PROJECT_SUBBUILD_DIR ${CMAKE_BINARY_DIR}/_deps/fast_tokenizer/sub-build)

  FetchContent_Declare(
    fast_tokenizer
    URL      https://github.com/PaddlePaddle/PaddleNLP/archive/refs/tags/v2.6.1.tar.gz
    URL_HASH SHA256=10e3489bc91e938c449a0448fa719e4536803ed6b1c1c95b3402430d6a8a221a
    PATCH_COMMAND git --git-dir=${EXTERNAL_PROJECT_SOURCE_DIR} apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/patches/fast_tokenizers.patch" &&
                  git --git-dir=${EXTERNAL_PROJECT_SOURCE_DIR} apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/patches/glog.patch" &&
                  git --git-dir=${EXTERNAL_PROJECT_SOURCE_DIR} apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/patches/icu.patch"
    SOURCE_DIR ${EXTERNAL_PROJECT_SOURCE_DIR}
    BINARY_DIR ${EXTERNAL_PROJECT_BINARY_DIR}
    SUBBUILD_DIR ${EXTERNAL_PROJECT_SUBBUILD_DIR}
  )

  FetchContent_GetProperties(fast_tokenizer)
  if(NOT fast_tokenizer_POPULATED)
    FetchContent_Populate(
      fast_tokenizer
    )
    set(EXTERNAL_OPTIONAL_ARGS
      -DCMAKE_POLICY_DEFAULT_CMP0057=NEW
      -DCMAKE_POLICY_DEFAULT_CMP0135=NEW)
    set(WITH_PYTHON OFF)
    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    add_subdirectory(${fast_tokenizer_SOURCE_DIR}/fast_tokenizer
                     ${CMAKE_CURRENT_BINARY_DIR}/fast_tokenizer
                     EXCLUDE_FROM_ALL
                    )
  endif()

  # variables used later
  set(FAST_TOKENIZER_INCS
    "${fast_tokenizer_SOURCE_DIR}/fast_tokenizer"
    "${CMAKE_BINARY_DIR}/third_party/dart/src/extern_dart/include/"
    "${CMAKE_BINARY_DIR}/third_party/json/src/extern_json/single_include/"
    "${CMAKE_BINARY_DIR}/third_party/install/re2/include/")
  set(FAST_TOKENIZER_LIBS core_tokenizers)
elseif(ENABLE_FAST_TOKENIZERS)
  if(WIN32 AND X86_64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-win-x64-1.0.2.zip
        URL_HASH SHA256=56470954014bdd3c8c8ad702d20f5f6aa5ab913bff92fd9c3c49ec6da31ff11d
      )
      ov_tokenizers_build_static_re2()
  elseif(LINUX AND X86_64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-x64-1.0.2.tgz
        URL_HASH SHA256=843a8299b55ef2e06ea50ba0d4ab4cb05b9e4cdb7cb8e29f3d55c494a1b7aecc
      )
  elseif(LINUX AND AARCH64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-linux-aarch64-1.0.2.tgz
        URL_HASH SHA256=fc16c51b24a954ae3d659e1b233ce15349eafc1e4c72710b51a4f12fb2c03033
      )
  elseif(APPLE AND X86_64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-x86_64-1.0.2.tgz
        # TODO: restore once https://github.com/PaddlePaddle/PaddleNLP/issues/7505 is fixed
        # URL_HASH SHA256=4c8123ad941b3e4325ef72f328db545e34d5eec2de3e2545e1ab8ebeeb5146a9
      )
  elseif(APPLE AND AARCH64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-osx-arm64-1.0.2.tgz
        URL_HASH SHA256=ffb0f16ec96b2f5dbdb681d00d74e932e273ec1c2108196d13f2fd28abc4d266
      )
  else()
    message(FATAL_ERROR "Platform ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR} does not have prebuilt Fast Tokenizer"
                        "Please, use -DBUILD_FAST_TOKENIZERS=ON cmake option to enable build from soures")
  endif()

  FetchContent_MakeAvailable(fast_tokenizer)

  # to allow find_library to work with conda-forge env
  set(_old_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${CMAKE_FIND_ROOT_PATH_MODE_LIBRARY})
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY NEVER)
  include("${fast_tokenizer_SOURCE_DIR}/FastTokenizer.cmake")
  set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ${_old_CMAKE_FIND_ROOT_PATH_MODE_LIBRARY})
  # since FastTokenizers.cmake overrides C++ standard, let's override it once again to required one
  ov_tokenizers_set_cxx_standard()

  if(WIN32 AND X86_64)
      # we use re2 library in regex_normalization operation, so have to add to this list
      # because prebuilt fast_tokenizers package does not provide this library
      list(APPEND FAST_TOKENIZER_LIBS re2)
  endif()
else()
  # in case if we don't build fast tokenizers, we have to include re2 explicitly
  ov_tokenizers_build_static_re2()
endif()

function(ov_tokenizers_link_fast_tokenizer TARGET_NAME)
  if(ENABLE_FAST_TOKENIZERS)
    target_include_directories(${TARGET_NAME} PRIVATE ${FAST_TOKENIZER_INCS})
    target_link_libraries(${TARGET_NAME} PRIVATE ${FAST_TOKENIZER_LIBS})
    target_compile_definitions(${TARGET_NAME} PRIVATE ENABLE_FAST_TOKENIZERS)
  else()
    message(FATAL_ERROR "ENABLE_FAST_TOKENIZERS is turned off. This function must not be called")
  endif()
endfunction()

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

  target_include_directories(${TARGET_NAME} PRIVATE ${pcre2_BINARY_DIR})
  target_link_libraries(${TARGET_NAME} PRIVATE pcre2-8)
  target_compile_definitions(${TARGET_NAME} PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
endfunction()

function(ov_tokenizers_link_re2 TARGET_NAME)
  if(NOT ENABLE_FAST_TOKENIZERS)
    target_link_libraries(${TARGET_NAME} PRIVATE re2)
  endif()
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
if(ENABLE_FAST_TOKENIZERS)
  ov_tokenizers_link_fast_tokenizer(${TARGET_NAME})
endif()
ov_tokenizers_link_pcre2(${TARGET_NAME})
ov_tokenizers_link_re2(${TARGET_NAME})

string(REPLACE " " ";" extra_flags "${c_cxx_flags} ${cxx_flags}")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_OPTIONS "${extra_flags}")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime openvino::threading)

#
# Post build steps to copy core_tokenizers dependencies
#

if(ENABLE_FAST_TOKENIZERS)
  if(BUILD_FAST_TOKENIZERS)
    set(fast_tokenezers_libs_dir "${CMAKE_BINARY_DIR}/third_party/icu/src/extern_icu/icu4c/bin64")
  else()
    set(fast_tokenezers_libs_dir "${fast_tokenizer_SOURCE_DIR}/third_party/lib")
    if(WIN32 AND X86_64)
      set(extra_libs "${fast_tokenizer_SOURCE_DIR}/lib/core_tokenizers.dll")
    elseif(LINUX)
      set(extra_libs "${fast_tokenizer_SOURCE_DIR}/lib/libcore_tokenizers.so")
    elseif(APPLE)
      set(extra_libs "${fast_tokenizer_SOURCE_DIR}/lib/libcore_tokenizers.dylib")
    endif()
  endif()

if(WIN32 AND X86_64)
  list(APPEND extra_libs "${fast_tokenezers_libs_dir}/icudt70.dll"
                         "${fast_tokenezers_libs_dir}/icuuc70${CMAKE_DEBUG_POSTFIX}.dll")
endif()

  if(extra_libs)
    # post build steps
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy ${extra_libs} $<TARGET_FILE_DIR:${TARGET_NAME}>)
  endif()
endif()

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

# - Windows: `<openvino_dir>\runtime\bin\intel64\Release\`
# - MacOS_x86: `<openvino_dir>/runtime/lib/intel64/Release`
# - MacOS_arm64: `<openvino_dir>/runtime/lib/arm64/Release/`
# - Linux_x86: `<openvino_dir>/runtime/lib/intel64/`
# - Linux_arm64: `<openvino_dir>/runtime/lib/aarch64/`
string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" OPENVINO_TOKENIZERS_INSTALL_DIR)
if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "amd64.*|x86_64.*|AMD64.*")
  set(OPENVINO_TOKENIZERS_INSTALL_DIR intel64)
elseif(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64.*|aarch64.*|AARCH64.*|ARM64.*)")
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
  set(OPENVINO_TOKENIZERS_INSTALL_DIR ${OPENVINO_TOKENIZERS_INSTALL_DIR}/${CMAKE_BUILD_TYPE})
endif()
set(OPENVINO_TOKENIZERS_INSTALL_BINDIR "runtime/bin/${OPENVINO_TOKENIZERS_INSTALL_DIR}" CACHE STRING "Destination for files installation of bin files - Windows dll")
set(OPENVINO_TOKENIZERS_INSTALL_LIBDIR "runtime/lib/${OPENVINO_TOKENIZERS_INSTALL_DIR}" CACHE STRING  "Destination for files installation of lib files")

# Installing the extension module
install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_LIBDIR} COMPONENT openvino_tokenizers
        RUNTIME DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_BINDIR} COMPONENT openvino_tokenizers)

if(BUILD_FAST_TOKENIZERS)
  install(TARGETS core_tokenizers
    LIBRARY DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_LIBDIR} COMPONENT openvino_tokenizers
    RUNTIME DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_BINDIR} COMPONENT openvino_tokenizers)
endif()

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
