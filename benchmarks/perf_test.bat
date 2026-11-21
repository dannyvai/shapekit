@echo off
REM GPU bundle adjustment self-test + re-alignment timing on projects\gpu_test. Log: benchmarks\perf_test.log
cd /d "%~dp0.."
echo === %date% %time% === > benchmarks\perf_test.log
uv run python -m tests.ba_selftest >> benchmarks\perf_test.log 2>&1
echo --- realign gpu_test (features + matches reused) --- >> benchmarks\perf_test.log
uv run python -m shapekit align projects\gpu_test --features aliked --reset >> benchmarks\perf_test.log 2>&1
echo exit %errorlevel% >> benchmarks\perf_test.log
