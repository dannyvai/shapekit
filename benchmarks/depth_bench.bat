@echo off
REM GPU plane-sweep depth kernel benchmark. Log: benchmarks\depth_bench.log
cd /d "%~dp0.."
uv run python -u -m benchmarks.depth_bench projects\gpu_test > benchmarks\depth_bench.log 2>&1
echo exit %errorlevel% >> benchmarks\depth_bench.log
