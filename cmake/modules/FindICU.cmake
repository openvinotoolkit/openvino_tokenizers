# Custom FindICU.cmake for ICU source tarball
set(ICU_SOURCE_DIR "" CACHE PATH "Path to extracted ICU source directory")
set(ICU_BUILD_DIR "" CACHE PATH "Path to build ICU from sources")
set(ICU_INSTALL_DIR "" CACHE PATH "Path to extracted ICU install directory")

# Ensure paths are provided
if (NOT ICU_SOURCE_DIR)
    message(FATAL_ERROR "ICU_SOURCE_DIR is required but not set.")
endif() 

# Create imported targets even if libraries are not available yet
if(NOT TARGET ICU::uc)
    add_library(ICU::uc UNKNOWN IMPORTED)
    # Set properties for targets, using placeholders for not-yet-built paths
    set_target_properties(ICU::uc PROPERTIES
        IMPORTED_LOCATION ${ICU_UC_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ICU_INCLUDE_DIRS}
    )
endif()
if(NOT TARGET ICU::i18n)
    add_library(ICU::i18n UNKNOWN IMPORTED)
    set_target_properties(ICU::i18n PROPERTIES
        IMPORTED_LOCATION ${ICU_I18N_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${ICU_INCLUDE_DIRS}
    )
endif()
if(NOT TARGET ICU::data)
    add_library(ICU::data UNKNOWN IMPORTED)
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
