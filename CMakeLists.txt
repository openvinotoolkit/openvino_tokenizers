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
        VERSION 2024.5.0.0
        DESCRIPTION "OpenVINO Tokenizers"
        HOMEPAGE_URL "https://github.com/openvinotoolkit/openvino_tokenizers"
        LANGUAGES CXX)

include(cmake/platforms.cmake)

option(BUILD_CPP_EXTENSION "Builds C++ extension for OpenVINO Tokenizers" ON)

if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(is_subproject ON)
endif()

if(NOT is_subproject AND DEFINED PY_BUILD_CMAKE_PACKAGE_VERSION AND NOT PY_BUILD_CMAKE_PACKAGE_VERSION EQUAL openvino_tokenizers_VERSION)
    message(FATAL_ERROR "openvino_tokenizers_VERSION (${openvino_tokenizers_VERSION}) is not equal to PY_BUILD_CMAKE_PACKAGE_VERSION (${PY_BUILD_CMAKE_PACKAGE_VERSION})")
endif()

if(BUILD_CPP_EXTENSION)
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
    find_package(OpenVINODeveloperPackage ${openvino_tokenizers_VERSION} QUIET
                 COMPONENTS Runtime Threading
                 PATHS "${OpenVINO_DIR}")
    if(NOT OpenVINODeveloperPackage_FOUND)
        find_package(OpenVINO ${openvino_tokenizers_VERSION} REQUIRED
                     COMPONENTS Runtime Threading
                     PATHS "${OpenVINO_DIR_PY}")
    endif()

    add_subdirectory(src)
endif()

# install python files

install(FILES "${openvino_tokenizers_SOURCE_DIR}/LICENSE"
              "${openvino_tokenizers_SOURCE_DIR}/third-party-programs.txt"
              "${openvino_tokenizers_SOURCE_DIR}/SECURITY.md"
        DESTINATION "${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.dist-info"
        COMPONENT openvino_tokenizers_licenses
        EXCLUDE_FROM_ALL)

configure_file("${openvino_tokenizers_SOURCE_DIR}/cmake/templates/__version__.py.in"
               "${openvino_tokenizers_BINARY_DIR}/python/__version__.py" @ONLY)
install(FILES "${openvino_tokenizers_BINARY_DIR}/python/__version__.py"
        DESTINATION "openvino_tokenizers"
        COMPONENT openvino_tokenizers_python
        EXCLUDE_FROM_ALL)
