# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.20.5)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(fast_tokenizer-populate NONE)



include(ExternalProject)
ExternalProject_Add(fast_tokenizer-populate
                     "UPDATE_DISCONNECTED" "False" "URL" "https://github.com/PaddlePaddle/PaddleNLP/archive/refs/tags/v2.6.1.tar.gz" "URL_HASH" "SHA256=10e3489bc91e938c449a0448fa719e4536803ed6b1c1c95b3402430d6a8a221a" "PATCH_COMMAND" "git" "--git-dir=C:/src/openvino_tokenizers_public/build/_deps/fast_tokenizer/src" "apply" "--ignore-whitespace" "C:/src/openvino_tokenizers_public/src/icu4c.patch"
                    SOURCE_DIR          "C:/src/openvino_tokenizers_public/build/_deps/fast_tokenizer/src"
                    BINARY_DIR          "C:/src/openvino_tokenizers_public/build/_deps/fast_tokenizer/build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)


