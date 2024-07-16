# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/src")
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/src")
endif()
file(MAKE_DIRECTORY
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/build"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/sub-build/fast_tokenizer-populate-prefix"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/sub-build/fast_tokenizer-populate-prefix/tmp"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/sub-build/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/sub-build/fast_tokenizer-populate-prefix/src"
  "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/sub-build/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/sub-build/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/rmikhail/src/openvino_tokenizers/.py-build-cmake_cache/cp311-cp311-linux_x86_64/_deps/fast_tokenizer/sub-build/fast_tokenizer-populate-prefix/src/fast_tokenizer-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
