@echo off
REM Automated GUI test on a copy of projects\gpu_test (screenshots in projects\gpu_test_guitest\selftest). Log: tests\gui_selftest.log
cd /d "%~dp0.."
uv run python -m tests.gui_selftest projects\gpu_test > tests\gui_selftest.log 2>&1
echo exit %errorlevel% >> tests\gui_selftest.log
