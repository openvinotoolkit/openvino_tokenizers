# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.15)

# Multi config generators such as Visual Studio ignore CMAKE_BUILD_TYPE. Multi config generators are configured with
# CMAKE_CONFIGURATION_TYPES, but limiting options in it completely removes such build options
get_property(GENERATOR_IS_MULTI_CONFIG_VAR GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(CMAKE_GENERATOR STREQUAL "Ninja Multi-Config")
    # 'Ninja Multi-Config' specific, see:
    # https://cmake.org/cmake/help/latest/variable/CMAKE_DEFAULT_BUILD_TYPE.html
    set(CMAKE_DEFAULT_BUILD_TYPE "Release" CACHE STRING "CMake default build type")
elseif(NOT GENERATOR_IS_MULTI_CONFIG_VAR)
    if(NOT DEFINED CMAKE_BUILD_TYPE)
        message(STATUS "CMAKE_BUILD_TYPE is not defined, 'Release' will be used")
    endif()
    # Setting CMAKE_BUILD_TYPE as CACHE must go before project(). Otherwise project() sets its value and set() doesn't take an effect
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel ...")
endif()

project(openvino_tokenizers)

option(CPACK_INCLUDE_TOPLEVEL_DIRECTORY "Boolean toggle to include/exclude top level directory." OFF)
if(WIN32)
    set(CPACK_GENERATOR "ZIP" CACHE STRING "List of CPack generators to use.")
endif()

include(cmake/platforms.cmake)

# Find OpenVINODeveloperPackage first to compile with SDL flags
find_package(OpenVINODeveloperPackage QUIET
             PATHS "${OpenVINO_DIR}")
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime OPTIONAL_COMPONENTS TensorFlow)
endif()

# set the project version using OpenVINO major, minor and patch versions
set(CMAKE_PROJECT_VERSION_TWEAK 0)
set(CMAKE_PROJECT_VERSION "${OpenVINO_VERSION}.${CMAKE_PROJECT_VERSION_TWEAK}")

add_subdirectory(src)
