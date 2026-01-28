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

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

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

if(NOT DEFINED CMAKE_DEBUG_POSTFIX AND (WIN32 OR APPLE))
  set(CMAKE_DEBUG_POSTFIX "d")
endif()

#
# Dependencies
#

include(FetchContent)

# This option must be defined early because it affects SentencePiece configuration
option(REGENERATE_PRECOMPILED_CHARSMAP "Regenerate the precompiled charsmap header (requires ICU at build time)" OFF)

FetchContent_Declare(
  sentencepiece
  URL      https://github.com/google/sentencepiece/releases/download/v0.2.1/sentencepiece-0.2.1.tar.gz
  URL_HASH SHA256=8138cec27c2f2282f4a34d9a016e3374cd40e5c6e9cb335063db66a0a3b71fad
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
  # NFKC compile requires ICU and is only needed for charsmap generation
  if(REGENERATE_PRECOMPILED_CHARSMAP)
    set(SPM_ENABLE_NFKC_COMPILE ON CACHE BOOL "Enable NFKC compile")
  else()
    set(SPM_ENABLE_NFKC_COMPILE OFF CACHE BOOL "Enable NFKC compile")
  endif()

  FetchContent_Populate(sentencepiece)

  # handle archives that add an extra top-level directory (e.g. .../sentencepiece/...)
  set(_sp_src "${sentencepiece_SOURCE_DIR}")
  if(NOT EXISTS "${_sp_src}/CMakeLists.txt")
    if(EXISTS "${_sp_src}/sentencepiece/CMakeLists.txt")
      set(_sp_src "${_sp_src}/sentencepiece")
    else()
      file(GLOB _sp_children RELATIVE "${sentencepiece_SOURCE_DIR}" "${sentencepiece_SOURCE_DIR}/*")
      foreach(_c ${_sp_children})
        if(IS_DIRECTORY "${sentencepiece_SOURCE_DIR}/${_c}" AND EXISTS "${sentencepiece_SOURCE_DIR}/${_c}/CMakeLists.txt")
          set(_sp_src "${sentencepiece_SOURCE_DIR}/${_c}")
          break()
        endif()
      endforeach()
    endif()
  endif()

  if(EXISTS "${_sp_src}/CMakeLists.txt")
    if(NOT _sp_src STREQUAL sentencepiece_SOURCE_DIR)
      message(STATUS "Adjusted sentencepiece source dir to: ${_sp_src}")
      # don't rely on changing the cached FetchContent variable; use _sp_src directly
    endif()
    add_subdirectory("${_sp_src}" "${sentencepiece_BINARY_DIR}" EXCLUDE_FROM_ALL)
  else()
    message(WARNING "SentencePiece source does not contain CMakeLists.txt; skipping add_subdirectory().")
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
        "${_sp_src}/src/builtin_pb" # for "sentencepiece_model.pb.h"
        "${_sp_src}/src" # for "normalizer.h"
        "${_sp_src}/third_party/protobuf-lite" # for "google/protobuf/port_def.inc"
        "${_sp_src}/third_party/" # for "absl/strings/string_view.h" and "darts_clone/darts.h"
        "${_sp_src}" # for "third_party/absl/strings/string_view.h"
        )
    else()
      target_include_directories(${TARGET_NAME} SYSTEM PRIVATE
        "${_sp_src}/src" # for "normalizer.h"
        "${_sp_src}" # for "third_party/absl/strings/string_view.h"
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

function(ov_tokenizers_link_pcre2 TARGET_NAME)
  FetchContent_Declare(
      pcre2
      URL https://github.com/PCRE2Project/pcre2/releases/download/pcre2-10.46/pcre2-10.46.zip
      URL_HASH SHA256=d872153b2d2338f7bc7b6e9bcc2f7f0c8a529c34fe48c538fea0b3a4e062f046
  )
  FetchContent_GetProperties(pcre2)
  if(NOT pcre2_POPULATED)
    FetchContent_Populate(pcre2)

    set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
    set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
    set(PCRE2_SUPPORT_JIT ON)
    set(PCRE2_STATIC_PIC ON)
    set(PCRE2_BUILD_TESTS OFF)
    set(PCRE2_BUILD_PCRE2GREP OFF)

    add_subdirectory(${pcre2_SOURCE_DIR} ${pcre2_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()

  target_include_directories(${TARGET_NAME} SYSTEM PRIVATE ${pcre2_BINARY_DIR})
  target_link_libraries(${TARGET_NAME} PRIVATE pcre2-8)
  target_compile_definitions(${TARGET_NAME} PRIVATE PCRE2_CODE_UNIT_WIDTH=8)
endfunction()

#
# Precompiled charsmap header
#
# The precompiled_charsmap.hpp header contains Unicode normalization tables that are
# pre-generated and committed to the repository. This eliminates the need to build
# and run the generator tool during normal builds, which significantly simplifies
# the build process and removes the ICU build-time dependency for most users.
#
# To regenerate the header (e.g., when updating Unicode tables or SentencePiece),
# set REGENERATE_PRECOMPILED_CHARSMAP=ON and ensure ICU is available.
#

set(PRECOMPILED_CHARSMAP_HEADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/precompiled_charsmap.hpp")

if(REGENERATE_PRECOMPILED_CHARSMAP)
  message(STATUS "REGENERATE_PRECOMPILED_CHARSMAP is ON - will build generator and regenerate charsmap header")
  
  # ICU is only needed for regenerating the charsmap header
  option(ENABLE_SYSTEM_ICU "Use system-installed ICU instead of building from source" OFF)
  
  if(ENABLE_SYSTEM_ICU)
    message(STATUS "Using system-installed ICU for charsmap generation.")
  else()
    message(STATUS "Building ICU from source for charsmap generation.")
    include(${PROJECT_SOURCE_DIR}/cmake/external/icu.cmake)
  endif()
  
  set(CHARSMAP_GENERATOR_NAME "generate_precompiled_charsmap")
  set(PRECOMPILED_CHARSMAP_HEADER_GENERATED "${CMAKE_CURRENT_BINARY_DIR}/precompiled_charsmap.hpp")

  # Build the generator tool as a host executable
  add_executable(${CHARSMAP_GENERATOR_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/generate_precompiled_charsmap.cpp)

  # The generator needs special configuration to access sentencepiece builder
  if(sentencepiece_FOUND)
    # Using system sentencepiece - link with both libraries
    foreach(sp_target sentencepiece sentencepiece_train)
      if(TARGET ${sp_target}-static)
        target_link_libraries(${CHARSMAP_GENERATOR_NAME} PRIVATE ${sp_target}-static)
      else()
        target_link_libraries(${CHARSMAP_GENERATOR_NAME} PRIVATE ${sp_target})
      endif()
    endforeach()
    target_link_libraries(${CHARSMAP_GENERATOR_NAME} PRIVATE absl::string_view absl::flat_hash_set)
  else()
    # Using built sentencepiece - need access to builder.h and config.h
    if(SPM_PROTOBUF_PROVIDER STREQUAL "internal")
      target_include_directories(${CHARSMAP_GENERATOR_NAME} SYSTEM PRIVATE
        "${_sp_src}/src/builtin_pb"
        "${_sp_src}/src"
        "${_sp_src}/third_party/protobuf-lite"
        "${_sp_src}/third_party/"
        "${_sp_src}"
        "${sentencepiece_BINARY_DIR}/src"  # For config.h
        "${sentencepiece_BINARY_DIR}"      # For config.h in root
      )
    else()
      target_include_directories(${CHARSMAP_GENERATOR_NAME} SYSTEM PRIVATE
        "${_sp_src}/src"
        "${_sp_src}"
        "${sentencepiece_BINARY_DIR}/src"
        "${sentencepiece_BINARY_DIR}"
      )
    endif()
    
    # Link with both sentencepiece libraries - the generator needs the builder from train library
    target_link_libraries(${CHARSMAP_GENERATOR_NAME} PRIVATE sentencepiece-static sentencepiece_train-static)
    
    if(ANDROID)
      target_link_libraries(${CHARSMAP_GENERATOR_NAME} PRIVATE log)
    endif()
  endif()

  # The generator needs ICU at build time to generate the maps
  find_package(ICU COMPONENTS i18n data uc REQUIRED)
  target_link_libraries(${CHARSMAP_GENERATOR_NAME} PRIVATE ICU::i18n ICU::data ICU::uc)

  # On Linux, ICU may need libdl for dynamic loading
  if(UNIX AND NOT APPLE AND NOT ANDROID)
    target_link_libraries(${CHARSMAP_GENERATOR_NAME} PRIVATE ${CMAKE_DL_LIBS})
  endif()

  # On Windows, copy ICU DLLs to the generator's directory so it can run
  if(WIN32 AND TARGET ${ICU_TARGET_NAME})
    if(GENERATOR_IS_MULTI_CONFIG_VAR)
      # Multi-config generator (Visual Studio)
      set(ICU_UC_DLL $<IF:$<CONFIG:Debug>,${ICU_UC_SHARED_LIB_DEBUG},${ICU_UC_SHARED_LIB_RELEASE}>)
      set(ICU_DATA_DLL $<IF:$<CONFIG:Debug>,${ICU_DATA_SHARED_LIB_DEBUG},${ICU_DATA_SHARED_LIB_RELEASE}>)
      set(ICU_I18N_DLL $<IF:$<CONFIG:Debug>,${ICU_I18N_SHARED_LIB_DEBUG},${ICU_I18N_SHARED_LIB_RELEASE}>)
    else()
      # Single-config generator (Ninja, NMake)
      if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set(ICU_UC_DLL "${ICU_UC_SHARED_LIB_DEBUG}")
        set(ICU_DATA_DLL "${ICU_DATA_SHARED_LIB_DEBUG}")
        set(ICU_I18N_DLL "${ICU_I18N_SHARED_LIB_DEBUG}")
      else()
        set(ICU_UC_DLL "${ICU_UC_SHARED_LIB_RELEASE}")
        set(ICU_DATA_DLL "${ICU_DATA_SHARED_LIB_RELEASE}")
        set(ICU_I18N_DLL "${ICU_I18N_SHARED_LIB_RELEASE}")
      endif()
    endif()
    
    add_custom_command(TARGET ${CHARSMAP_GENERATOR_NAME} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ICU_UC_DLL} $<TARGET_FILE_DIR:${CHARSMAP_GENERATOR_NAME}>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ICU_DATA_DLL} $<TARGET_FILE_DIR:${CHARSMAP_GENERATOR_NAME}>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${ICU_I18N_DLL} $<TARGET_FILE_DIR:${CHARSMAP_GENERATOR_NAME}>
      COMMENT "Copying ICU DLLs for generator executable"
    )
  endif()

  # Ensure sentencepiece is built before the generator
  if(NOT sentencepiece_FOUND)
    if(TARGET sentencepiece-static)
      add_dependencies(${CHARSMAP_GENERATOR_NAME} sentencepiece-static)
    endif()
    if(TARGET sentencepiece_train-static)
      add_dependencies(${CHARSMAP_GENERATOR_NAME} sentencepiece_train-static)
    endif()
  endif()

  # If building ICU from source, ensure it's built before sentencepiece_train
  # (sentencepiece_train uses ICU for normalization during charsmap generation)
  if(TARGET ${ICU_TARGET_NAME})
    if(TARGET sentencepiece_train-static)
      add_dependencies(sentencepiece_train-static ${ICU_TARGET_NAME})
    endif()
    add_dependencies(${CHARSMAP_GENERATOR_NAME} ${ICU_TARGET_NAME})
  endif()

  # Run the generator to create the precompiled charsmap header
  add_custom_command(
      OUTPUT ${PRECOMPILED_CHARSMAP_HEADER_GENERATED}
      COMMAND ${CHARSMAP_GENERATOR_NAME} ${PRECOMPILED_CHARSMAP_HEADER_GENERATED}
      DEPENDS ${CHARSMAP_GENERATOR_NAME}
      COMMENT "Generating precompiled charsmap header"
      VERBATIM
  )

  # Create a custom target to ensure the header is generated
  add_custom_target(generate_charsmap_header DEPENDS ${PRECOMPILED_CHARSMAP_HEADER_GENERATED})

  # Add a convenience target to update the source file
  add_custom_target(update_precompiled_charsmap
      COMMAND ${CMAKE_COMMAND} -E copy ${PRECOMPILED_CHARSMAP_HEADER_GENERATED} ${PRECOMPILED_CHARSMAP_HEADER_SOURCE}
      DEPENDS ${PRECOMPILED_CHARSMAP_HEADER_GENERATED}
      COMMENT "Updating precompiled_charsmap.hpp in source directory"
  )

  set(PRECOMPILED_CHARSMAP_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
else()
  # Use the pre-committed header from the source directory
  if(NOT EXISTS "${PRECOMPILED_CHARSMAP_HEADER_SOURCE}")
    message(FATAL_ERROR "Pre-committed precompiled_charsmap.hpp not found at ${PRECOMPILED_CHARSMAP_HEADER_SOURCE}. "
                        "Either commit this file or set REGENERATE_PRECOMPILED_CHARSMAP=ON to generate it.")
  endif()
  message(STATUS "Using pre-committed precompiled_charsmap.hpp from source directory")
  set(PRECOMPILED_CHARSMAP_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

#
# Build library
#

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
list(REMOVE_ITEM SRCS ${CMAKE_CURRENT_SOURCE_DIR}/generate_precompiled_charsmap.cpp)

add_library(${TARGET_NAME} SHARED ${SRCS})

# Ensure the precompiled charsmap header is generated before building the library (only when regenerating)
if(REGENERATE_PRECOMPILED_CHARSMAP)
  add_dependencies(${TARGET_NAME} generate_charsmap_header)
endif()

# Add the appropriate directory to include path for the precompiled charsmap header
target_include_directories(${TARGET_NAME} PRIVATE ${PRECOMPILED_CHARSMAP_INCLUDE_DIR})

if(WIN32)
  add_vs_version_resource(${TARGET_NAME})
endif()

#
# Target include dirs, link libraries and other properties
#

ov_tokenizers_link_sentencepiece(${TARGET_NAME})
ov_tokenizers_link_pcre2(${TARGET_NAME})

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
  if(CPACK_GENERATOR STREQUAL "NPM")
    # to find libopenvino.so when installing from npm
    list(APPEND rpaths "$ORIGIN/../../openvino-node/bin")
    # to find libopenvino.so when installing from source
    list(APPEND rpaths "$ORIGIN/../node_modules/openvino-node/bin")
  endif()
elseif(APPLE)
  # to find libcore_tokenizer.dylib in the same folder
  set(rpaths "@loader_path")
  if(DEFINED PY_BUILD_CMAKE_PACKAGE_NAME)
    # in case we build pip package, we need to refer to libopenvino.dylib from 'openvino' package
    list(APPEND rpaths "@loader_path/../../openvino/libs")
  endif()
  if(CPACK_GENERATOR STREQUAL "NPM")
    # to find libopenvino.dylib when installing from npm
    list(APPEND rpaths "@loader_path/../../openvino-node/bin")
    # to find libopenvino.dylib when installing from source
    list(APPEND rpaths "@loader_path/../node_modules/openvino-node/bin")
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
