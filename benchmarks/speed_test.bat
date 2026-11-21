@echo off
REM Optimised vs original code paths on copies of projects\gpu_test: timing + identical-output check. Log: benchmarks\speed_test.log
cd /d "%~dp0.."
uv run python -u -m benchmarks.speed_selftest projects\gpu_test > benchmarks\speed_test.log 2>&1
echo exit %errorlevel% >> benchmarks\speed_test.log
