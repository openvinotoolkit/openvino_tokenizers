@REM This is a generated file.
@echo off
setlocal
SET srcdir="C:\src\openvino_tokenizers_public\build\_deps\pcre2-src"
# The next line was replaced by the following one after a user comment.
# SET pcre2test="$<TARGET_FILE:pcre2test>"
SET pcre2test="C:\src\openvino_tokenizers_public\build\_deps\pcre2-build\pcre2test.exe"
if not [%CMAKE_CONFIG_TYPE%]==[] SET pcre2test="C:\src\openvino_tokenizers_public\build\_deps\pcre2-build\%CMAKE_CONFIG_TYPE%\pcre2test.exe"
call %srcdir%\RunTest.Bat
if errorlevel 1 exit /b 1
echo RunTest.bat tests successfully completed
