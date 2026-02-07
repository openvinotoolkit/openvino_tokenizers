# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Custom FindICU.cmake for charsmap generation
# Creates imported targets for ICU libraries (Release only)

if(NOT TARGET ICU::uc)
    add_library(ICU::uc STATIC IMPORTED)
    set_target_properties(ICU::uc PROPERTIES
        IMPORTED_LOCATION ${ICU_UC_LIB}
    )
endif()
if(NOT TARGET ICU::i18n)
    add_library(ICU::i18n STATIC IMPORTED)
    set_target_properties(ICU::i18n PROPERTIES
        IMPORTED_LOCATION ${ICU_I18N_LIB}
    )
endif()
if(NOT TARGET ICU::data)
    add_library(ICU::data STATIC IMPORTED)
    set_target_properties(ICU::data PROPERTIES
        IMPORTED_LOCATION ${ICU_DATA_LIB}
    )
endif()

message(STATUS "ICU_INCLUDE_DIRS: ${ICU_INCLUDE_DIRS}")
message(STATUS "ICU_UC_LIB: ${ICU_UC_LIB}")
message(STATUS "ICU_I18N_LIB: ${ICU_I18N_LIB}")
message(STATUS "ICU_DATA_LIB: ${ICU_DATA_LIB}")

# Standard message for package finding
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ICU DEFAULT_MSG ICU_INCLUDE_DIRS ICU_LIBRARIES)

