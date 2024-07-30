# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.20.5)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(sentencepiece-populate NONE)



include(ExternalProject)
ExternalProject_Add(sentencepiece-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/google/sentencepiece/archive/refs/tags/v0.2.0.tar.gz" "URL_HASH" "SHA256=9970f0a0afee1648890293321665e5b2efa04eaec9f1671fcf8048f456f5bb86"
                    SOURCE_DIR          "C:/src/openvino_tokenizers_public/build/_deps/sentencepiece-src"
                    BINARY_DIR          "C:/src/openvino_tokenizers_public/build/_deps/sentencepiece-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


