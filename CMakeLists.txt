# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.15)

project(openvino_tokenizers)

include(cmake/platforms.cmake)

# Find OpenVINODeveloperPackage first to compile with SDL flags
find_package(OpenVINODeveloperPackage QUIET
             PATHS "${OpenVINO_DIR}")
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime OPTIONAL_COMPONENTS TensorFlow)
endif()

# wa for conda channel in 2024.1 release
set(OpenVINO_Frontend_TensorFlow_FOUND OFF)

# set the project version using OpenVINO major, minor and patch versions
set(CMAKE_PROJECT_VERSION_TWEAK 0)
set(CMAKE_PROJECT_VERSION "${OpenVINO_VERSION}.${CMAKE_PROJECT_VERSION_TWEAK}")

add_subdirectory(src)
