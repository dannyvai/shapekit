@echo off
REM Depth maps with mixed portrait/landscape neighbours (GPU vs CPU) + same-size identity check. Log: tests\mixed_depth_test.log
cd /d "%~dp0.."
uv run python -u -m tests.mixed_depth_test projects\gpu_test projects\area_d > tests\mixed_depth_test.log 2>&1
echo exit %errorlevel% >> tests\mixed_depth_test.log
