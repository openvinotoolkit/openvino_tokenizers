# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# ICU build configuration for charsmap generation
#
# This builds ICU only for the generator tool, not for shipping.
# Only Release builds are supported since ICU is only used at build time.
#

include(FetchContent)

set(ICU_VERSION "70")
set(ICU_TARGET_NAME "icu_external")
set(ICU_URL https://github.com/unicode-org/icu/releases/download/release-70-1/icu4c-70_1-src.tgz)
set(ICU_URL_HASH SHA256=8d205428c17bf13bb535300669ed28b338a157b1c01ae66d31d0d3e2d47c3fd5)

set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/_deps/icu)
set(ICU_SOURCE_DIR  ${THIRD_PARTY_PATH}/icu-src)
set(ICU_INSTALL_DIR ${THIRD_PARTY_PATH}/icu-install)
set(ICU_BINARY_DIR ${THIRD_PARTY_PATH}/icu-build)

set(ICU_INCLUDE_DIRS "${ICU_INSTALL_DIR}/include")

# Define build artifacts (Release only)

if(WIN32)
    set(ICU_INSTALL_LIB_SUBDIR "lib64")
    set(ICU_INSTALL_BIN_SUBDIR "bin64")
    set(ICU_UC_LIB_NAME "icuuc")
    set(ICU_I18N_LIB_NAME "icuin")
    set(ICU_DATA_LIB_NAME "icudt")
    set(ICU_UC_SHARED_LIB_NAME "${ICU_UC_LIB_NAME}${ICU_VERSION}")
    set(ICU_I18N_SHARED_LIB_NAME "${ICU_I18N_LIB_NAME}${ICU_VERSION}")
    set(ICU_DATA_SHARED_LIB_NAME "${ICU_DATA_LIB_NAME}${ICU_VERSION}")
    set(ICU_STATIC_SUFFIX ".lib")
    set(ICU_SHARED_SUFFIX ".dll")
else()
    set(ICU_INSTALL_LIB_SUBDIR "lib")
    set(ICU_INSTALL_BIN_SUBDIR "lib")
    set(ICU_UC_LIB_NAME "icuuc")
    set(ICU_I18N_LIB_NAME "icui18n")
    set(ICU_DATA_LIB_NAME "icudata")
    set(ICU_UC_SHARED_LIB_NAME ${ICU_UC_LIB_NAME})
    set(ICU_I18N_SHARED_LIB_NAME ${ICU_I18N_LIB_NAME})
    set(ICU_DATA_SHARED_LIB_NAME ${ICU_DATA_LIB_NAME})
    set(ICU_STATIC_SUFFIX ".a")
    set(ICU_SHARED_SUFFIX ".so")

    # Calculate the number of cores
    execute_process(COMMAND nproc
      OUTPUT_VARIABLE ICU_JOB_POOL_SIZE
      OUTPUT_STRIP_TRAILING_WHITESPACE)
endif()

# Set library paths (Release only)
set(ICU_LIB_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_LIB_SUBDIR}")
set(ICU_BIN_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_BIN_SUBDIR}")

set(ICU_UC_LIB "${ICU_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${ICU_UC_LIB_NAME}${ICU_STATIC_SUFFIX}")
set(ICU_I18N_LIB "${ICU_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${ICU_I18N_LIB_NAME}${ICU_STATIC_SUFFIX}")
set(ICU_DATA_LIB "${ICU_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${ICU_DATA_LIB_NAME}${ICU_STATIC_SUFFIX}")

set(ICU_UC_SHARED_LIB "${ICU_BIN_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${ICU_UC_SHARED_LIB_NAME}${ICU_SHARED_SUFFIX}")
set(ICU_I18N_SHARED_LIB "${ICU_BIN_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${ICU_I18N_SHARED_LIB_NAME}${ICU_SHARED_SUFFIX}")
set(ICU_DATA_SHARED_LIB "${ICU_BIN_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}${ICU_DATA_SHARED_LIB_NAME}${ICU_SHARED_SUFFIX}")

set(ICU_LIBRARIES ${ICU_UC_LIB} ${ICU_I18N_LIB} ${ICU_DATA_LIB})

# Compile & link flags

set(ICU_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(ICU_C_FLAGS "${CMAKE_C_FLAGS}")
set(ICU_LINKER_FLAGS "")

if(NOT WIN32)
  set(ICU_CXX_FLAGS "${ICU_CXX_FLAGS} -fPIC -Wno-deprecated-declarations")
  set(ICU_C_FLAGS "${ICU_C_FLAGS} -fPIC -Wno-deprecated-declarations")
  if(CMAKE_DL_LIBS)
    set(ICU_LINKER_FLAGS "-l${CMAKE_DL_LIBS}")
  endif()
endif()

# Propagate _GLIBCXX_USE_CXX11_ABI from OpenVINO
get_target_property(OPENVINO_RUNTIME_COMPILE_DEFINITIONS openvino::runtime INTERFACE_COMPILE_DEFINITIONS)
if(OPENVINO_RUNTIME_COMPILE_DEFINITIONS)
  foreach(def IN LISTS OPENVINO_RUNTIME_COMPILE_DEFINITIONS)
    set(ICU_CXX_FLAGS "${ICU_CXX_FLAGS} -D${def}")
    set(ICU_C_FLAGS "${ICU_C_FLAGS} -D${def}")
  endforeach()
endif()

message(STATUS "ICU_CXX_FLAGS: ${ICU_CXX_FLAGS}")
message(STATUS "ICU_C_FLAGS: ${ICU_C_FLAGS}")
message(STATUS "ICU_LINKER_FLAGS: ${ICU_LINKER_FLAGS}")

# Build ICU

include(ExternalProject)

if(CMAKE_C_COMPILER_LAUNCHER)
  set(c_prefix "${CMAKE_C_COMPILER_LAUNCHER} ")
endif()
if(CMAKE_CXX_COMPILER_LAUNCHER)
  set(cxx_prefix "${CMAKE_CXX_COMPILER_LAUNCHER} ")
endif()

if(WIN32)
  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    SOURCE_DIR ${ICU_SOURCE_DIR}
    BINARY_DIR ${ICU_BINARY_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    PATCH_COMMAND powershell -Command "(Get-ChildItem -Path <SOURCE_DIR>/source -Recurse -File -Filter \"*.vcxproj\" | ForEach-Object { (Get-Content $_.FullName) -replace '<DebugInformationFormat>EditAndContinue</DebugInformationFormat>', '<DebugInformationFormat>ProgramDatabase</DebugInformationFormat>' | Set-Content $_.FullName })" &&
                  powershell -Command "(Get-ChildItem -Path <SOURCE_DIR>/source -Recurse -File -Filter \"*.mak\" | ForEach-Object { (Get-Content $_.FullName) -replace 'py\\s*-3', 'python' | Set-Content $_.FullName })"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ${CMAKE_COMMAND} -E env CL=${ICU_CXX_FLAGS} LINK=${ICU_LINKER_FLAGS} msbuild ${ICU_SOURCE_DIR}\\source\\allinone\\allinone.sln /p:Configuration=Release /p:Platform=x64 /t:i18n /t:uconv /t:makedata
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/include ${ICU_INSTALL_DIR}/include &&
                    ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/lib64 ${ICU_LIB_DIR} &&
                    ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/bin64 ${ICU_BIN_DIR}
    BUILD_BYPRODUCTS ${ICU_LIBRARIES})
else()
  # Compiler settings
  if(NOT APPLE)
    set(CC_ENV "${c_prefix}${CMAKE_C_COMPILER}")
    set(CXX_ENV "${cxx_prefix}${CMAKE_CXX_COMPILER}")
  else()
    set(CC_ENV "${c_prefix}cc")
    set(CXX_ENV "${cxx_prefix}c++")
  endif()

  # Unset environment variables that could interfere with cross-compilation
  set(HOST_ENV_UNSET "")
  foreach(var SDKTARGETSYSROOT CONFIG_SITE OECORE_NATIVE_SYSROOT OECORE_TARGET_SYSROOT
              OECORE_ACLOCAL_OPTS OECORE_BASELIB OECORE_TARGET_ARCH OECORE_TARGET_OS CC CXX
              CPP AS LD GDB STRIP RANLIB OBJCOPY OBJDUMP READELF AR NM M4 TARGET_PREFIX
              CONFIGURE_FLAGS CFLAGS CXXFLAGS LDFLAGS CPPFLAGS KCFLAGS OECORE_DISTRO_VERSION
              OECORE_SDK_VERSION ARCH CROSS_COMPILE OE_CMAKE_TOOLCHAIN_FILE OPENSSL_CONF
              OE_CMAKE_FIND_LIBRARY_CUSTOM_LIB_SUFFIX PKG_CONFIG_SYSROOT_DIR PKG_CONFIG_PATH)
    if(DEFINED ENV{${var}})
      list(APPEND HOST_ENV_UNSET --unset=${var})
    endif()
  endforeach()

  set(ICU_CONFIGURE_FLAGS
    --prefix=${ICU_INSTALL_DIR}
    --includedir=${ICU_INCLUDE_DIRS}
    --enable-static
    --enable-release
    --enable-tools
    --disable-rpath
    --disable-shared
    --disable-tests
    --disable-samples
    --disable-extras
    --disable-icuio
    --disable-draft
    --disable-icu-config
    --disable-layoutex
    --disable-layout)

  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    SOURCE_DIR ${ICU_SOURCE_DIR}
    BINARY_DIR ${ICU_BINARY_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    CONFIGURE_COMMAND ${CMAKE_COMMAND} -E env ${HOST_ENV_UNSET} CC=${CC_ENV} CXX=${CXX_ENV} CFLAGS=${ICU_C_FLAGS} CXXFLAGS=${ICU_CXX_FLAGS} LDFLAGS=${ICU_LINKER_FLAGS} ${ICU_SOURCE_DIR}/source/configure ${ICU_CONFIGURE_FLAGS}
    BUILD_COMMAND make -j${ICU_JOB_POOL_SIZE}
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${ICU_LIBRARIES})
endif()

# Use custom FindICU module to create imported targets
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules" ${CMAKE_MODULE_PATH})
