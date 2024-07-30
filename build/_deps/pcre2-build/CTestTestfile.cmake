# CMake generated Testfile for 
# Source directory: C:/src/openvino_tokenizers_public/build/_deps/pcre2-src
# Build directory: C:/src/openvino_tokenizers_public/build/_deps/pcre2-build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
if("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Dd][Ee][Bb][Uu][Gg])$")
  add_test(pcre2_test_bat "pcre2_test.bat")
  set_tests_properties(pcre2_test_bat PROPERTIES  PASS_REGULAR_EXPRESSION "RunTest\\.bat tests successfully completed" _BACKTRACE_TRIPLES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;1077;ADD_TEST;C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;0;")
elseif("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
  add_test(pcre2_test_bat "pcre2_test.bat")
  set_tests_properties(pcre2_test_bat PROPERTIES  PASS_REGULAR_EXPRESSION "RunTest\\.bat tests successfully completed" _BACKTRACE_TRIPLES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;1077;ADD_TEST;C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;0;")
elseif("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Mm][Ii][Nn][Ss][Ii][Zz][Ee][Rr][Ee][Ll])$")
  add_test(pcre2_test_bat "pcre2_test.bat")
  set_tests_properties(pcre2_test_bat PROPERTIES  PASS_REGULAR_EXPRESSION "RunTest\\.bat tests successfully completed" _BACKTRACE_TRIPLES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;1077;ADD_TEST;C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;0;")
elseif("${CTEST_CONFIGURATION_TYPE}" MATCHES "^([Rr][Ee][Ll][Ww][Ii][Tt][Hh][Dd][Ee][Bb][Ii][Nn][Ff][Oo])$")
  add_test(pcre2_test_bat "pcre2_test.bat")
  set_tests_properties(pcre2_test_bat PROPERTIES  PASS_REGULAR_EXPRESSION "RunTest\\.bat tests successfully completed" _BACKTRACE_TRIPLES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;1077;ADD_TEST;C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;0;")
else()
  add_test(pcre2_test_bat NOT_AVAILABLE)
endif()
add_test(pcre2posix_test "pcre2posix_test")
set_tests_properties(pcre2posix_test PROPERTIES  _BACKTRACE_TRIPLES "C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;1099;ADD_TEST;C:/src/openvino_tokenizers_public/build/_deps/pcre2-src/CMakeLists.txt;0;")
