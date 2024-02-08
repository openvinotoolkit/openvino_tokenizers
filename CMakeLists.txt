# Copyright (C) 2022-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

cmake_minimum_required(VERSION 3.15)

project(openvino_tokenizers)

include(cmake/platforms.cmake)

# Find OpenVINODeveloperPackage first to compile with SDL flags
find_package(OpenVINODeveloperPackage QUIET
             PATHS "${OpenVINO_DIR}")
if(NOT OpenVINODeveloperPackage_FOUND)
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

# set the project version using OpenVINO major, minor and patch versions
set(CMAKE_PROJECT_VERSION_TWEAK 0)
set(CMAKE_PROJECT_VERSION "${OpenVINO_VERSION}.${CMAKE_PROJECT_VERSION_TWEAK}")

include(GNUInstallDirs)

# setting RPATH / LC_RPATH depending on platform
if(LINUX)
  # to find libcore_tokenizer.so in the same folder
  set(rpaths "$ORIGIN")
elseif(APPLE)
  # to find libcore_tokenizer.dylib in the same folder
  set(rpaths "@loader_path")
  if(DEFINED SKBUILD)
    # in case we build pip package, we need to refer to libopenvino.dylib from 'openvino' package
    list(APPEND rpaths "@loader_path/../../openvino/libs")
  endif()
endif()

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()

add_subdirectory(src)


