# Install script for directory: C:/src/openvino_tokenizers_public/build/_deps/pcre2-src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files/openvino_tokenizers")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Debug/pcre2-8-staticd.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Release/pcre2-8-static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/MinSizeRel/pcre2-8-static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/RelWithDebInfo/pcre2-8-static.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Debug/pcre2-posix-staticd.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Release/pcre2-posix-static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/MinSizeRel/pcre2-posix-static.lib")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/RelWithDebInfo/pcre2-posix-static.lib")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Debug/pcre2grep.exe")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Release/pcre2grep.exe")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/MinSizeRel/pcre2grep.exe")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/RelWithDebInfo/pcre2grep.exe")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Debug/pcre2test.exe")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/Release/pcre2test.exe")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/MinSizeRel/pcre2test.exe")
  elseif("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/RelWithDebInfo/pcre2test.exe")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/pkgconfig" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/libpcre2-posix.pc"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/libpcre2-8.pc"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE PERMISSIONS OWNER_WRITE OWNER_READ OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/pcre2-config")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/pcre2.h"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/src/pcre2posix.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/cmake" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/cmake/pcre2-config.cmake"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-build/cmake/pcre2-config-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man1" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2-config.1"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2grep.1"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2test.1"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/man/man3" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_callout_enumerate.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_code_copy.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_code_copy_with_tables.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_code_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_compile.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_compile_context_copy.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_compile_context_create.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_compile_context_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_config.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_convert_context_copy.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_convert_context_create.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_convert_context_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_converted_pattern_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_dfa_match.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_general_context_copy.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_general_context_create.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_general_context_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_get_error_message.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_get_mark.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_get_match_data_heapframes_size.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_get_match_data_size.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_get_ovector_count.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_get_ovector_pointer.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_get_startchar.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_jit_compile.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_jit_free_unused_memory.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_jit_match.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_jit_stack_assign.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_jit_stack_create.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_jit_stack_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_maketables.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_maketables_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_match.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_match_context_copy.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_match_context_create.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_match_context_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_match_data_create.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_match_data_create_from_pattern.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_match_data_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_pattern_convert.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_pattern_info.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_serialize_decode.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_serialize_encode.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_serialize_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_serialize_get_number_of_codes.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_bsr.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_callout.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_character_tables.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_compile_extra_options.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_compile_recursion_guard.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_depth_limit.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_glob_escape.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_glob_separator.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_heap_limit.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_match_limit.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_max_pattern_compiled_length.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_max_pattern_length.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_max_varlookbehind.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_newline.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_offset_limit.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_parens_nest_limit.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_recursion_limit.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_recursion_memory_management.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_set_substitute_callout.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substitute.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_copy_byname.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_copy_bynumber.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_get_byname.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_get_bynumber.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_length_byname.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_length_bynumber.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_list_free.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_list_get.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_nametable_scan.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2_substring_number_from_name.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2api.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2build.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2callout.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2compat.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2convert.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2demo.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2jit.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2limits.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2matching.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2partial.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2pattern.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2perform.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2posix.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2sample.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2serialize.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2syntax.3"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/pcre2unicode.3"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/doc/pcre2/html" TYPE FILE FILES
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/index.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2-config.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_callout_enumerate.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_code_copy.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_code_copy_with_tables.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_code_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_compile.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_compile_context_copy.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_compile_context_create.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_compile_context_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_config.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_convert_context_copy.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_convert_context_create.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_convert_context_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_converted_pattern_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_dfa_match.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_general_context_copy.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_general_context_create.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_general_context_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_get_error_message.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_get_mark.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_get_match_data_heapframes_size.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_get_match_data_size.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_get_ovector_count.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_get_ovector_pointer.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_get_startchar.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_jit_compile.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_jit_free_unused_memory.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_jit_match.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_jit_stack_assign.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_jit_stack_create.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_jit_stack_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_maketables.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_maketables_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_match.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_match_context_copy.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_match_context_create.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_match_context_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_match_data_create.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_match_data_create_from_pattern.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_match_data_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_pattern_convert.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_pattern_info.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_serialize_decode.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_serialize_encode.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_serialize_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_serialize_get_number_of_codes.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_bsr.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_callout.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_character_tables.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_compile_extra_options.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_compile_recursion_guard.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_depth_limit.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_glob_escape.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_glob_separator.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_heap_limit.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_match_limit.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_max_pattern_compiled_length.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_max_pattern_length.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_max_varlookbehind.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_newline.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_offset_limit.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_parens_nest_limit.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_recursion_limit.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_recursion_memory_management.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_set_substitute_callout.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substitute.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_copy_byname.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_copy_bynumber.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_get_byname.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_get_bynumber.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_length_byname.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_length_bynumber.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_list_free.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_list_get.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_nametable_scan.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2_substring_number_from_name.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2api.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2build.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2callout.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2compat.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2convert.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2demo.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2grep.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2jit.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2limits.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2matching.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2partial.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2pattern.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2perform.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2posix.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2sample.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2serialize.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2syntax.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2test.html"
    "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/doc/html/pcre2unicode.html"
    )
endif()

