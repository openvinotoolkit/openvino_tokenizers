# Custom FindICU.cmake for ICU source tarball
set(ICU_SOURCE_DIR "" CACHE PATH "Path to extracted ICU source directory")
set(ICU_BUILD_DIR "" CACHE PATH "Path to build ICU from sources")
set(ICU_INSTALL_DIR "" CACHE PATH "Path to extracted ICU install directory")

# Ensure paths are provided
if (NOT ICU_SOURCE_DIR)
    message(FATAL_ERROR "ICU_SOURCE_DIR is required but not set.")
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

if(ICU_INSTALL_DIR)
    set(ICU_INCLUDE_DIRS "${ICU_INSTALL_DIR}/include")
    set(ICU_STATIC_LIB_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_LIB_SUBDIR}")
    set(ICU_SHARED_LIB_DIR "${ICU_INSTALL_DIR}/${ICU_INSTALL_BIN_SUBDIR}")
endif()

if(ICU_STATIC)
    set(TYPE "STATIC")
else()
    set(TYPE "SHARED")
endif()

# Create imported targets even if libraries are not available yet
if(NOT TARGET ICU::uc)
    add_library(ICU::uc UNKNOWN IMPORTED)
    # Set properties for targets, using placeholders for not-yet-built paths
    set(ICU_UC_LIB "${ICU_${TYPE}_LIB_DIR}/${ICU_${TYPE}_PREFIX}icuuc.${ICU_${TYPE}_SUFFIX}")
    set_target_properties(ICU::uc PROPERTIES
        IMPORTED_LOCATION ${ICU_UC_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ICU_INCLUDE_DIRS}
    )
endif()
if(NOT TARGET ICU::i18n)
    add_library(ICU::i18n UNKNOWN IMPORTED)
    set(ICU_I18N_LIB "${ICU_${TYPE}_LIB_DIR}/${ICU_${TYPE}_PREFIX}icuin.${ICU_${TYPE}_SUFFIX}")
    set_target_properties(ICU::i18n PROPERTIES
        IMPORTED_LOCATION ${ICU_I18N_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ICU_INCLUDE_DIRS}
    )
endif()
if(NOT TARGET ICU::data)
    add_library(ICU::data UNKNOWN IMPORTED)
    set(ICU_DATA_LIB "${ICU_${TYPE}_LIB_DIR}/${ICU_${TYPE}_PREFIX}icudt.${ICU_${TYPE}_SUFFIX}")
    set_target_properties(ICU::data PROPERTIES
        IMPORTED_LOCATION ${ICU_DATA_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ICU_INCLUDE_DIRS}
    )
endif()

# Check if ICU is already built by verifying library paths
find_path(ICU_INCLUDE_DIRS unicode/utypes.h PATHS ${ICU_INCLUDE_DIRS} NO_DEFAULT_PATH)
find_library(ICU_UC_LIB NAMES icuuc${ICU_SUFFIX} PATHS ${ICU_${TYPE}_LIB_DIR} NO_DEFAULT_PATH)
find_library(ICU_I18N_LIB NAMES icuin${ICU_SUFFIX} PATHS ${ICU_${TYPE}_LIB_DIR} NO_DEFAULT_PATH)
find_library(ICU_DATA_LIB NAMES icudata${ICU_SUFFIX} PATHS ${ICU_${TYPE}_LIB_DIR} NO_DEFAULT_PATH)

list(APPEND ICU_LIBRARIES ${ICU_UC_LIB})
list(APPEND ICU_LIBRARIES ${ICU_I18N_LIB})
list(APPEND ICU_LIBRARIES ${ICU_DATA_LIB})

message(STATUS "ICU_ROOT: ${ICU_ROOT}")
message(STATUS "ICU_INCLUDE_DIRS: ${ICU_INCLUDE_DIRS}")
message(STATUS "ICU_UC_LIBRARY: ${ICU_UC_LIBRARY}")
message(STATUS "ICU_I18N_LIB: ${ICU_I18N_LIB}")
message(STATUS "ICU_DATA_LIB: ${ICU_DATA_LIB}")

# Standard message for package finding
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ICU DEFAULT_MSG ICU_INCLUDE_DIRS ICU_LIBRARIES)
