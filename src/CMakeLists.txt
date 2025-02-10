# Copyright (C) 2018-2025 Intel Corporation
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

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

if(NOT DEFINED CMAKE_DEBUG_POSTFIX AND WIN32)
  set(CMAKE_DEBUG_POSTFIX "d")
endif()

#
# Dependencies
#

include(FetchContent)

option(ENABLE_SYSTEM_ICU "Enables use of system ICU" OFF)

if(ENABLE_SYSTEM_ICU) 
  message(STATUS "Using system-installed ICU.")
else()
  message(STATUS "ICU is not found and will be built from source.")
  include(${PROJECT_SOURCE_DIR}/cmake/external/icu.cmake)
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
  
  if(TARGET ${ICU_TARGET_NAME})
    foreach(sp_target sentencepiece sentencepiece_train)
      if(TARGET ${sp_target}-static)
        set(sp_target ${sp_target}-static)  
        add_dependencies(${sp_target} ${ICU_TARGET_NAME})
      endif()
    endforeach()
  endif()

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
    if(SPM_PROTOBUF_PROVIDER STREQUAL "internal")
      if(SPM_ABSL_PROVIDER STREQUAL "package")
        message(FATAL_ERROR "When 'SPM_PROTOBUF_PROVIDER' is 'package', 'SPM_ABSL_PROVIDER' must have the same value")
      endif()

      target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        "${sentencepiece_SOURCE_DIR}/src/builtin_pb" # for "sentencepiece_model.pb.h"
        "${sentencepiece_SOURCE_DIR}/src" # for "normalizer.h"
        "${sentencepiece_SOURCE_DIR}/third_party/protobuf-lite" # for "google/protobuf/port_def.inc"
        "${sentencepiece_SOURCE_DIR}/third_party/" # for "absl/strings/string_view.h"
        "${sentencepiece_SOURCE_DIR}" # for "third_party/absl/strings/string_view.h"
        )
    else()
      target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        "${sentencepiece_SOURCE_DIR}/src" # for "normalizer.h"
        "${sentencepiece_SOURCE_DIR}" # for "third_party/absl/strings/string_view.h"
        "${sentencepiece_BINARY_DIR}/src" # for "sentencepiece_model.pb.h"
        )
      endif()

    foreach(sp_target sentencepiece-static sentencepiece_train-static)
      if(CMAKE_CL_64)
        target_compile_definitions(${sp_target} PRIVATE _CRT_SECURE_NO_WARNINGS _SCL_SECURE_NO_WARNINGS)
      endif()
      # to propogate _GLIBCXX_USE_CXX11_ABI value
      target_compile_definitions(${sp_target} PUBLIC $<TARGET_PROPERTY:openvino::runtime,INTERFACE_COMPILE_DEFINITIONS>)
      target_link_libraries(${TARGET_NAME} PRIVATE ${sp_target})
    endforeach()

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

# sentencepiece uses ICU libraries and is linked as INTERFACE, which is why we need to manually link tokenizers with ICU
find_package(ICU COMPONENTS i18n data uc REQUIRED)
target_link_libraries(${TARGET_NAME} PRIVATE ICU::i18n ICU::data ICU::uc)

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
    set(OPENVINO_TOKENIZERS_INSTALL_DIR ${OPENVINO_TOKENIZERS_INSTALL_DIR}/${BUILD_TYPE})
  endif()

  set(OPENVINO_TOKENIZERS_INSTALL_BINDIR "runtime/bin/${OPENVINO_TOKENIZERS_INSTALL_DIR}" CACHE STRING "Destination for files installation of bin files - Windows dll")
  set(OPENVINO_TOKENIZERS_INSTALL_LIBDIR "runtime/lib/${OPENVINO_TOKENIZERS_INSTALL_DIR}" CACHE STRING  "Destination for files installation of lib files")
endif()

# Installing the extension module
install(TARGETS ${TARGET_NAME}
        LIBRARY DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_LIBDIR} COMPONENT openvino_tokenizers
        RUNTIME DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_BINDIR} COMPONENT openvino_tokenizers)

if(WIN32)
  # post build
  if(GENERATOR_IS_MULTI_CONFIG_VAR)
    # Select the appropriate ICU UC shared library based on the build configuration (Debug or Release)
    set(ICU_UC_SHARED_LIB $<IF:$<CONFIG:Debug>,${ICU_UC_SHARED_LIB_DEBUG},${ICU_UC_SHARED_LIB_RELEASE}>)
    set(ICU_DATA_SHARED_LIB $<IF:$<CONFIG:Debug>,${ICU_DATA_SHARED_LIB_DEBUG},${ICU_DATA_SHARED_LIB_RELEASE}>)
  else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      set(ICU_UC_SHARED_LIB "${ICU_UC_SHARED_LIB_DEBUG}")
      set(ICU_DATA_SHARED_LIB "${ICU_DATA_SHARED_LIB_DEBUG}")
    else()
      set(ICU_UC_SHARED_LIB "${ICU_UC_SHARED_LIB_RELEASE}")
      set(ICU_DATA_SHARED_LIB "${ICU_DATA_SHARED_LIB_RELEASE}")
    endif()
  endif()
  
  add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ICU_UC_SHARED_LIB} $<TARGET_FILE_DIR:${TARGET_NAME}>
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ICU_DATA_SHARED_LIB} $<TARGET_FILE_DIR:${TARGET_NAME}>
  )
  
  install(
    FILES ${ICU_UC_SHARED_LIB_RELEASE} ${ICU_DATA_SHARED_LIB_RELEASE}
    DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_BINDIR}
    CONFIGURATIONS Release RelWithDebInfo MinSizeRel
    COMPONENT openvino_tokenizers
  )  
  install(
    FILES ${ICU_UC_SHARED_LIB_DEBUG} ${ICU_DATA_SHARED_LIB_DEBUG}
    DESTINATION ${OPENVINO_TOKENIZERS_INSTALL_BINDIR}
    CONFIGURATIONS Debug
    COMPONENT openvino_tokenizers
  )
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
