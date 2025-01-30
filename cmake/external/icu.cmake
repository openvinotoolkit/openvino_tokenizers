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

if(NOT WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -std=c++11")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
endif()

if(OPENVINO_RUNTIME_COMPILE_DEFINITIONS)
    foreach(def ${OPENVINO_RUNTIME_COMPILE_DEFINITIONS})
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${def}")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D${def}")
    endforeach()
endif()

set(HOST_ENV_CMAKE ${CMAKE_COMMAND} -E env
        CFLAGS=${CMAKE_C_FLAGS}
        CXXFLAGS=${CMAKE_CXX_FLAGS}
        LDFLAGS=${CMAKE_MODULE_LINKER_FLAGS}
)

if(GENERATOR_IS_MULTI_CONFIG_VAR)
  set(ICU_CONFIGURE_FLAGS $<$<CONFIG:Debug>:"--enable-debug">$<$<CONFIG:Release>:"--enable-release">)
  set(ICU_BUILD_TYPE $<CONFIG>)
else()
  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ICU_CONFIGURE_FLAGS "--enable-debug")
  else()
    set(ICU_CONFIGURE_FLAGS "--enable-release")
  endif()
  set(ICU_BUILD_TYPE ${CMAKE_BUILD_TYPE})
endif()

if(WIN32)
    set(ICU_SHARED_PREFIX "")
    set(ICU_STATIC_PREFIX "")
    set(ICU_SHARED_SUFFIX ".dll")
    set(ICU_STATIC_SUFFIX ".lib")
    set(ICU_INSTALL_LIB_SUBDIR "lib64")
    set(ICU_INSTALL_BIN_SUBDIR "bin64")
    set(ICU_UC_LIB_NAME "icuuc")
    set(ICU_I18N_LIB_NAME "icuin")
    set(ICU_DATA_LIB_NAME "icudt")
    set(ICU_UC_SHARED_LIB_NAME "${ICU_UC_LIB_NAME}${ICU_VERSION}")
    set(ICU_I18N_SHARED_LIB_NAME "${ICU_I18N_LIB_NAME}${ICU_VERSION}")
    set(ICU_DATA_SHARED_LIB_NAME "${ICU_DATA_LIB_NAME}${ICU_VERSION}")
else()
    set(ICU_SHARED_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
    set(ICU_STATIC_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(ICU_SHARED_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(ICU_STATIC_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(ICU_INSTALL_LIB_SUBDIR "lib")
    set(ICU_INSTALL_BIN_SUBDIR "lib")
    set(ICU_UC_LIB_NAME "icuuc")
    set(ICU_I18N_LIB_NAME "icui18n")
    set(ICU_DATA_LIB_NAME "icudata")
    set(ICU_UC_SHARED_LIB_NAME ${ICU_UC_LIB_NAME})
    set(ICU_I18N_SHARED_LIB_NAME ${ICU_I18N_LIB_NAME})
    set(ICU_DATA_SHARED_LIB_NAME ${ICU_DATA_LIB_NAME})
    
    # Calculate the number of cores using CMake
    execute_process(COMMAND nproc OUTPUT_VARIABLE CMAKE_JOB_POOL_SIZE)
    string(STRIP ${CMAKE_JOB_POOL_SIZE} CMAKE_JOB_POOL_SIZE)
endif()

set(ICU_INCLUDE_DIRS "${ICU_INSTALL_DIR}/include")

set(ICU_STATIC_LIB_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_LIB_SUBDIR}")
set(ICU_SHARED_LIB_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_BIN_SUBDIR}")

foreach(build_type RELEASE DEBUG)
  foreach(icu_target UC I18N DATA)
    if(icu_target STREQUAL "DATA")
      set(lib_postfix ${CMAKE_RELEASE_POSTFIX})
    else()
      set(lib_postfix ${CMAKE_${build_type}_POSTFIX})
    endif()
    
    set(ICU_${icu_target}_LIB_${build_type} "${ICU_STATIC_LIB_DIR}/${ICU_STATIC_PREFIX}${ICU_${icu_target}_LIB_NAME}${lib_postfix}${ICU_STATIC_SUFFIX}")
    set(ICU_${icu_target}_SHARED_LIB_${build_type} "${ICU_SHARED_LIB_DIR}/${ICU_SHARED_PREFIX}${ICU_${icu_target}_SHARED_LIB_NAME}${lib_postfix}${ICU_SHARED_SUFFIX}")
    list(APPEND ICU_LIBRARIES_${build_type} ${ICU_${icu_target}_LIB_${build_type}})
    list(APPEND ICU_SHARED_LIBRARIES_${build_type} ${ICU_${icu_target}_LIB_${build_type}})
  endforeach()
endforeach()

include(ExternalProject)

if(WIN32)
  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    SOURCE_DIR ${ICU_SOURCE_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    CONFIGURE_COMMAND msbuild ${ICU_SOURCE_DIR}\\source\\allinone\\allinone.sln /p:Configuration=${ICU_BUILD_TYPE} /p:Platform=x64 /t:i18n /t:uconv /t:makedata 
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/include ${ICU_INSTALL_DIR}/include && 
                    ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/lib64 ${ICU_INSTALL_DIR}/lib64 &&
                    ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/bin64 ${ICU_INSTALL_DIR}/bin64
    BUILD_BYPRODUCTS ${ICU_LIBRARIES_RELEASE}
  )
elseif(APPLE)
  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    SOURCE_DIR ${ICU_SOURCE_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    CONFIGURE_COMMAND ${HOST_ENV_CMAKE} ${ICU_SOURCE_DIR}/source/runConfigureICU MacOSX --prefix ${ICU_INSTALL_DIR}
                      ${ICU_CONFIGURE_FLAGS} 
                      --enable-static
                      --enable-rpath
                      --disable-shared
                      --disable-tests
                      --disable-samples
                      --disable-extras
                      --disable-icuio
                      --disable-draft
                      --disable-icu-config
    BUILD_COMMAND make -j${CMAKE_JOB_POOL_SIZE} 
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${ICU_LIBRARIES_RELEASE}
  )
else()
  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    SOURCE_DIR ${ICU_SOURCE_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    CONFIGURE_COMMAND ${HOST_ENV_CMAKE} ${ICU_SOURCE_DIR}/source/runConfigureICU Linux --prefix ${ICU_INSTALL_DIR} 
                      ${ICU_CONFIGURE_FLAGS}
                      --enable-static
                      --enable-rpath
                      --disable-shared
                      --disable-tests
                      --disable-samples
                      --disable-extras
                      --disable-icuio
                      --disable-draft
                      --disable-icu-config
    BUILD_COMMAND make -j${CMAKE_JOB_POOL_SIZE} 
    INSTALL_COMMAND make install
    BUILD_BYPRODUCTS ${ICU_LIBRARIES_RELEASE}
  )
endif()

# using custom FindICU module
list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")