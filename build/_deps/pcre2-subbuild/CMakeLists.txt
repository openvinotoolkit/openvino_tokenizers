# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.20.5)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(pcre2-populate NONE)



include(ExternalProject)
ExternalProject_Add(pcre2-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/PCRE2Project/pcre2/archive/refs/tags/pcre2-10.44.zip" "URL_HASH" "SHA256=2d87bd1700bd1993ddea7c56aad2b0373ac2b3d52d9cc78842a6d061ffaf0925"
                    SOURCE_DIR          "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src"
                    BINARY_DIR          "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


