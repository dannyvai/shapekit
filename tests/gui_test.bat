@echo off
REM Opens the GUI on projects\gpu_test. Log: tests\gui_test.log
cd /d "%~dp0.."
echo === %date% %time% === > tests\gui_test.log
uv run python -m shapekit gui projects\gpu_test >> tests\gui_test.log 2>&1
echo GUI exited with %errorlevel% >> tests\gui_test.log
