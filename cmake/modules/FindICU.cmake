# Custom FindICU.cmake

# Create imported targets even if libraries are not available yet
if(NOT TARGET ICU::uc)
    add_library(ICU::uc UNKNOWN IMPORTED)
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

message(STATUS "ICU_INCLUDE_DIRS: ${ICU_INCLUDE_DIRS}")
message(STATUS "ICU_UC_LIB: ${ICU_UC_LIB}")
message(STATUS "ICU_I18N_LIB: ${ICU_I18N_LIB}")
message(STATUS "ICU_DATA_LIB: ${ICU_DATA_LIB}")

# Standard message for package finding
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ICU DEFAULT_MSG ICU_INCLUDE_DIRS ICU_LIBRARIES)
