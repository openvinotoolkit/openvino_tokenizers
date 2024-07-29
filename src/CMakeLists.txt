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

ov_tokenizers_set_cxx_standard()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WIN32)
  # Set the runtime library globally
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# openvino::runtime exports _GLIBCXX_USE_CXX11_ABI=0 on CentOS7.
# It needs to be propagated to every library openvino_tokenizers links with.
# That prohibits linkage with prebuilt libraries because they aren't compiled with _GLIBCXX_USE_CXX11_ABI=0.
get_directory_property(OPENVINO_RUNTIME_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
include(CMakeDependentOption)
if("_GLIBCXX_USE_CXX11_ABI=0" IN_LIST OPENVINO_RUNTIME_COMPILE_DEFINITIONS)
  set(USE_ABI0 ON CACHE BOOL "Set -D_GLIBCXX_USE_CXX11_ABI to 0 for fast_tokenizers")
endif()

option(BUILD_FAST_TOKENIZERS "Compile core_tokenizers instead of downloading prebuilt library. The option is forced to ON if _GLIBCXX_USE_CXX11_ABI=0" ${USE_ABI0})

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
  endif()
endfunction()

if(BUILD_FAST_TOKENIZERS)
  set(WITH_PYTHON OFF CACHE BOOL "")

  FetchContent_Declare(hiberlite
    GIT_REPOSITORY https://github.com/PaddlePaddle/PaddleNLP
    GIT_TAG v2.6.1
    PATCH_COMMAND git apply --ignore-whitespace "${CMAKE_CURRENT_LIST_DIR}/icu4c.patch"
  )

  FetchContent_GetProperties(fast_tokenizer)
  if(NOT fast_tokenizer_POPULATED)
    FetchContent_Populate(
      fast_tokenizer
    )
    set(WITH_PYTHON OFF CACHE BOOL "Disable Python API for fast_tokenizer")
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
else()
  if(WIN32 AND X86_64)
      FetchContent_Declare(
        fast_tokenizer
        URL      https://bj.bcebos.com/paddlenlp/fast_tokenizer/fast_tokenizer-win-x64-1.0.2.zip
        URL_HASH SHA256=56470954014bdd3c8c8ad702d20f5f6aa5ab913bff92fd9c3c49ec6da31ff11d
      )
      FetchContent_Declare(
        re2
        URL      https://github.com/google/re2/archive/refs/tags/2022-04-01.tar.gz
        URL_HASH SHA256=1ae8ccfdb1066a731bba6ee0881baad5efd2cd661acd9569b689f2586e1a50e9
      )
      function(ov_tokenizers_build_static_re2)
        set(BUILD_SHARED_LIBS OFF)
        FetchContent_GetProperties(re2)
        if(NOT re2_POPULATED)
          FetchContent_Populate(re2)
          add_subdirectory(${re2_SOURCE_DIR} ${re2_BINARY_DIR} EXCLUDE_FROM_ALL)
        endif()
      endfunction()
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
endif()

function(ov_tokenizers_link_fast_tokenizer)
  target_include_directories(${TARGET_NAME} PRIVATE ${FAST_TOKENIZER_INCS})
  target_link_libraries(${TARGET_NAME} PRIVATE ${FAST_TOKENIZER_LIBS})
endfunction()

#
# Build library
#

file(GLOB SRC *.cpp)
if(NOT OpenVINO_Frontend_TensorFlow_FOUND)
  list(REMOVE_ITEM SRC ${CMAKE_CURRENT_SOURCE_DIR}/tensorflow_translators.cpp)
endif()

add_library(${TARGET_NAME} SHARED ${SRC})

#
# Target include dirs, link libraries and other properties
#

ov_tokenizers_link_sentencepiece(${TARGET_NAME})
ov_tokenizers_link_fast_tokenizer(${TARGET_NAME})

string(REPLACE " " ";" extra_flags "${c_cxx_flags} ${cxx_flags}")
set_target_properties(${TARGET_NAME} PROPERTIES COMPILE_OPTIONS "${extra_flags}")

target_compile_definitions(${TARGET_NAME} PRIVATE IMPLEMENT_OPENVINO_EXTENSION_API)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime)

# Fetch PCRE2
include(FetchContent)
FetchContent_Declare(
    PCRE2
    URL https://github.com/PCRE2Project/pcre2/archive/refs/tags/pcre2-10.44.zip
    URL_HASH SHA256=2d87bd1700bd1993ddea7c56aad2b0373ac2b3d52d9cc78842a6d061ffaf0925
)
FetchContent_MakeAvailable(PCRE2)

target_include_directories(${TARGET_NAME} PRIVATE ${PCRE2_BINARY_DIR})
target_link_libraries(${TARGET_NAME} PRIVATE pcre2-8)
target_compile_definitions(${TARGET_NAME} PRIVATE PCRE2_CODE_UNIT_WIDTH=8)

if(OpenVINO_Frontend_TensorFlow_FOUND)
  target_link_libraries(${TARGET_NAME} PRIVATE openvino::frontend::tensorflow)
  target_compile_definitions(${TARGET_NAME} PRIVATE OpenVINO_Frontend_TensorFlow_FOUND)
endif()

#
# Post build steps to copy core_tokenizers dependencies
#

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
                         "${fast_tokenezers_libs_dir}/icuuc70.dll")
endif()

if(extra_libs)
  # post build steps
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${extra_libs} $<TARGET_FILE_DIR:${TARGET_NAME}>)
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
if(MSVC OR APPLE)
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
              "${openvino_tokenizers_SOURCE_DIR}/SECURITY.md"
        DESTINATION "${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.dist-info"
        COMPONENT openvino_tokenizers_licenses
        EXCLUDE_FROM_ALL)

install(FILES "${openvino_tokenizers_SOURCE_DIR}/LICENSE"
              "${openvino_tokenizers_SOURCE_DIR}/third-party-programs.txt"
              "${openvino_tokenizers_SOURCE_DIR}/README.md"
        DESTINATION "docs/openvino_tokenizers"
        COMPONENT openvino_tokenizers_docs
        EXCLUDE_FROM_ALL)

configure_file("${openvino_tokenizers_SOURCE_DIR}/cmake/templates/__version__.py.in"
               "${openvino_tokenizers_BINARY_DIR}/python/__version__.py" @ONLY)
install(FILES "${openvino_tokenizers_BINARY_DIR}/python/__version__.py"
        DESTINATION "openvino_tokenizers"
        COMPONENT openvino_tokenizers_python
        EXCLUDE_FROM_ALL)

#
# Cpack configuration
#

set(CPACK_COMPONENTS_ALL openvino_tokenizers openvino_tokenizers_docs)

# Uniform outcome in all repos - all repos will not create top level directory
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)

include (CPack)
