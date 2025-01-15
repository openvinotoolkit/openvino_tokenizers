include(FetchContent)

set(THIRD_PARTY_PATH ${CMAKE_BINARY_DIR}/_deps/icu)
set(ICU_SOURCE_DIR  ${THIRD_PARTY_PATH}/icu-src)
set(ICU_BINARY_DIR  ${THIRD_PARTY_PATH}/icu-build)
SET(ICU_INSTALL_DIR ${THIRD_PARTY_PATH}/icu-install)

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

set(FETCHCONTENT_QUIET FALSE)
# Fetch and build ICU
FetchContent_Declare(
    ICU
    URL https://github.com/unicode-org/icu/archive/refs/tags/release-70-1.tar.gz
    URL_HASH SHA256=f30d670bdc03ba999638a2d2511952ab94adf204d0e14898666f2e0cacb7fef1
    SOURCE_DIR ${ICU_SOURCE_DIR}
    BINARY_DIR ${ICU_BINARY_DIR}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_MakeAvailable(ICU)

if(NOT ICU_POPULATED)
    # Configure the ICU build
    message(STATUS "Configuring ICU...")
    execute_process(
        COMMAND ${ICU_SOURCE_DIR}/icu4c/source/runConfigureICU Linux --prefix ${ICU_INSTALL_DIR} ${ICU_CONFIGURE_FLAGS}
            --disable-tests
            --disable-samples
            --disable-tools
            --disable-extras 
            --disable-icuio
            --disable-draft
        WORKING_DIRECTORY ${ICU_BINARY_DIR}
    )
    message(STATUS "Building ICU...")
    execute_process(
        COMMAND make -j${CMAKE_JOB_POOL_SIZE}
        WORKING_DIRECTORY ${ICU_BINARY_DIR}
    )
    message(STATUS "Installing ICU...")
    execute_process(
        COMMAND make install
        WORKING_DIRECTORY ${ICU_BINARY_DIR}
    )
endif()
# Manually set ICU include and library directories
set(ICU_ROOT ${ICU_INSTALL_DIR})

if(WIN32)
    set(SHARED_LIB_EXT "*.dll")
elseif(APPLE)
    set(SHARED_LIB_EXT "*.dylib")
else()
    set(SHARED_LIB_EXT "*.so")
endif()

install(
    DIRECTORY ${ICU_INSTALL_DIR}/lib/
    DESTINATION $<TARGET_FILE_DIR:${TARGET_NAME}>
    FILES_MATCHING PATTERN "${SHARED_LIB_EXT}"
)