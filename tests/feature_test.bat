@echo off
REM GPU rasteriser / texture visibility + SALAD retrieval self-test. Log: tests\feature_test.log
cd /d "%~dp0.."
uv run python -m tests.feature_selftest projects\gpu_test projects\area_d > tests\feature_test.log 2>&1
echo exit %errorlevel% >> tests\feature_test.log
