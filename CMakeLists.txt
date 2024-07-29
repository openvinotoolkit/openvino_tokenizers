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

project(openvino_tokenizers
        VERSION 2024.3.0.0
        DESCRIPTION "OpenVINO Tokenizers"
        HOMEPAGE_URL "https://github.com/openvinotoolkit/openvino_tokenizers"
        LANGUAGES CXX)

        # Check if the build type is Debug
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Set compiler flags for Debug build
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -g")
endif()

include(cmake/platforms.cmake)

# Looking for OpenVINO in the python distribution. It doesn't work for cross-compiling build
if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 REQUIRED)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "from openvino.utils import get_cmake_path; print(get_cmake_path(), end='')"
        OUTPUT_VARIABLE OpenVINO_DIR_PY
        ERROR_QUIET
    )
endif()

# Find OpenVINODeveloperPackage first to compile with SDL flags
find_package(OpenVINODeveloperPackage QUIET PATHS "${OpenVINO_DIR}")
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime OPTIONAL_COMPONENTS TensorFlow PATHS "${OpenVINO_DIR_PY}")
endif()

if(DEFINED PY_BUILD_CMAKE_PACKAGE_VERSION AND NOT PY_BUILD_CMAKE_PACKAGE_VERSION EQUAL CMAKE_PROJECT_VERSION)
    message(FATAL_ERROR "CMAKE_PROJECT_VERSION (${CMAKE_PROJECT_VERSION}) is not equal to PY_BUILD_CMAKE_PACKAGE_VERSION (${PY_BUILD_CMAKE_PACKAGE_VERSION})")
endif()

add_subdirectory(src)
