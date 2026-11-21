@echo off
rem Runs the roadmap self-test, the GUI self-test, then builds the portable ShapeKit folder (~30-45 min).
rem Logs: tests\roadmap_selftest.log, tests\gui_selftest.log, build_portable.log
cd /d "%~dp0"
title ShapeKit: tests + portable build
echo [1/3] roadmap self-test ...
call "%~dp0roadmap_selftest.bat"
echo [2/3] GUI self-test ...
call "%~dp0gui_selftest.bat"
echo [3/3] portable build ...
call "%~dp0..\build_portable.bat"
echo Finished. You can close this window.
