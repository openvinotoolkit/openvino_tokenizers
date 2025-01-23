include(FetchContent)

set(ICU_TARGET_NAME "icu_external")
set(ICU_URL https://github.com/unicode-org/icu/releases/download/release-70-1/icu4c-70_1-src.tgz)
set(ICU_URL_HASH SHA256=8d205428c17bf13bb535300669ed28b338a157b1c01ae66d31d0d3e2d47c3fd5)

set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/_deps/icu)
set(ICU_SOURCE_DIR  ${THIRD_PARTY_PATH}/icu-src CACHE PATH "Path to extracted ICU source directory")
set(ICU_BINARY_DIR  ${THIRD_PARTY_PATH}/icu-build CACHE PATH "Path to extracted ICU binary directory")
set(ICU_INSTALL_DIR ${THIRD_PARTY_PATH}/icu-install CACHE PATH "Path to extracted ICU install directory")

set(ICU_STATIC TRUE)

set(HOST_ENV_CMAKE ${CMAKE_COMMAND} -E env
        CC=${CMAKE_C_COMPILER}
        CXX=${CMAKE_CXX_COMPILER}
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

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(ICU_SUFFIX "d")  # Debug suffix used by ICU
else()
    set(ICU_SUFFIX "")
endif()

if(WIN32)
    set(ICU_SHARED_PREFIX "lib")
    set(ICU_STATIC_PREFIX "")
    set(ICU_SHARED_SUFFIX "dll")
    set(ICU_STATIC_SUFFIX "lib")
    set(ICU_INSTALL_LIB_SUBDIR "lib64")
    set(ICU_INSTALL_BIN_SUBDIR "bin64")
elseif()
    set(ICU_SHARED_PREFIX ${CMAKE_SHARED_LIBRARY_PREFIX})
    set(ICU_STATIC_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
    set(ICU_SHARED_SUFFIX ${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(ICU_STATIC_SUFFIX ${CMAKE_STATIC_LIBRARY_SUFFIX})
    set(ICU_INSTALL_LIB_SUBDIR "lib")
    set(ICU_INSTALL_BIN_SUBDIR "lib")
endif()

if(ICU_STATIC)
    set(TYPE "STATIC")
else()
    set(TYPE "SHARED")
endif()

if(ICU_INSTALL_DIR)
    set(ICU_INCLUDE_DIRS "${ICU_INSTALL_DIR}/include")
    set(ICU_STATIC_LIB_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_LIB_SUBDIR}")
    set(ICU_SHARED_LIB_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_BIN_SUBDIR}")
endif()

set(ICU_UC_LIB "${ICU_${TYPE}_LIB_DIR}/${ICU_${TYPE}_PREFIX}icuuc.${ICU_${TYPE}_SUFFIX}")
set(ICU_I18N_LIB "${ICU_${TYPE}_LIB_DIR}/${ICU_${TYPE}_PREFIX}icuin.${ICU_${TYPE}_SUFFIX}")
set(ICU_DATA_LIB "${ICU_${TYPE}_LIB_DIR}/${ICU_${TYPE}_PREFIX}icudt.${ICU_${TYPE}_SUFFIX}")

include(ExternalProject)

if(WIN32)
  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    SOURCE_DIR ${ICU_SOURCE_DIR}
    BINARY_DIR ${ICU_BUILD_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    CONFIGURE_COMMAND msbuild ${ICU_SOURCE_DIR}\\source\\allinone\\allinone.sln /p:OutDir=${ICU_BINARY_DIR} /p:Configuration=${ICU_BUILD_TYPE} /p:Platform=x64 /p:SkipUWP=true /t:i18n
    BUILD_COMMAND ""
    INSTALL_COMMAND ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/include ${ICU_INSTALL_DIR}/include && 
                    ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/lib64 ${ICU_INSTALL_DIR}/lib64 &&
                    ${CMAKE_COMMAND} -E copy_directory ${ICU_SOURCE_DIR}/bin64 ${ICU_INSTALL_DIR}/bin64
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    BUILD_BYPRODUCTS ${ICU_UC_LIB} ${ICU_I18N_LIB} ${ICU_DATA_LIB}
  )
elseif(APPLE)
  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    CONFIGURE_COMMAND ${HOST_ENV_CMAKE} ${ICU_SOURCE_DIR}/source/runConfigureICU MacOSX --prefix ${ICU_INSTALL_DIR} ${ICU_CONFIGURE_FLAGS} --disable-tests --disable-samples --disable-tools --disable-extras --disable-icuio --disable-draft --disable-icu-config
    BUILD_COMMAND make -j${CMAKE_JOB_POOL_SIZE}
    INSTALL_COMMAND make install
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    BUILD_BYPRODUCTS ${ICU_UC_LIB} ${ICU_I18N_LIB} ${ICU_DATA_LIB}
  )
else()
  ExternalProject_Add(
    ${ICU_TARGET_NAME}
    URL ${ICU_URL}
    URL_HASH ${ICU_URL_HASH}
    PREFIX ${THIRD_PARTY_PATH}
    SOURCE_DIR ${ICU_SOURCE_DIR}
    BINARY_DIR ${ICU_BUILD_DIR}
    INSTALL_DIR ${ICU_INSTALL_DIR}
    CONFIGURE_COMMAND ${HOST_ENV_CMAKE} ${ICU_SOURCE_DIR}/source/runConfigureICU Linux --prefix ${ICU_INSTALL_DIR} ${ICU_CONFIGURE_FLAGS} --disable-tests --disable-samples --disable-tools --disable-extras --disable-icuio --disable-draft --disable-icu-config
    BUILD_COMMAND make -j${CMAKE_JOB_POOL_SIZE}
    INSTALL_COMMAND make install
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    BUILD_BYPRODUCTS ${ICU_UC_LIB} ${ICU_I18N_LIB} ${ICU_DATA_LIB}
  )
endif()