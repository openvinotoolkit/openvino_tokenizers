# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# Custom FindICU.cmake
# Create imported targets even if libraries are not available yet

if(NOT TARGET ICU::uc)
    add_library(ICU::uc UNKNOWN IMPORTED)
    set_target_properties(ICU::uc PROPERTIES
        IMPORTED_LOCATION ${ICU_UC_LIB_RELEASE}
        IMPORTED_LOCATION_DEBUG ${ICU_UC_LIB_DEBUG}
    )
endif()
if(NOT TARGET ICU::i18n)
    add_library(ICU::i18n UNKNOWN IMPORTED)
    set_target_properties(ICU::i18n PROPERTIES
        IMPORTED_LOCATION ${ICU_I18N_LIB_RELEASE}
        IMPORTED_LOCATION_DEBUG ${ICU_I18N_LIB_DEBUG}
    )
endif()
if(NOT TARGET ICU::data)
    add_library(ICU::data UNKNOWN IMPORTED)
    set_target_properties(ICU::data PROPERTIES
        IMPORTED_LOCATION ${ICU_DATA_LIB_RELEASE}
        IMPORTED_LOCATION_DEBUG ${ICU_DATA_LIB_DEBUG}
    )
endif()

message(STATUS "ICU_INCLUDE_DIRS: ${ICU_INCLUDE_DIRS}")
message(STATUS "ICU_UC_LIB_RELEASE: ${ICU_UC_LIB_RELEASE}")
message(STATUS "ICU_I18N_LIB_RELEASE: ${ICU_I18N_LIB_RELEASE}")
message(STATUS "ICU_DATA_LIB_RELEASE: ${ICU_DATA_LIB_RELEASE}")
message(STATUS "ICU_UC_LIB_DEBUG: ${ICU_UC_LIB_DEBUG}")
message(STATUS "ICU_I18N_LIB_DEBUG: ${ICU_I18N_LIB_DEBUG}")
message(STATUS "ICU_DATA_LIB_DEBUG: ${ICU_DATA_LIB_DEBUG}")

# Standard message for package finding
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ICU DEFAULT_MSG ICU_INCLUDE_DIRS ICU_LIBRARIES_RELEASE ICU_LIBRARIES_DEBUG)
