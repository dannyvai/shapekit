@echo off
REM PatchMatch, fisheye, masks, mesh tools, multiband texture, chunks, mixed-precision BA (on copies of projects\gpu_test). Log: tests\roadmap_selftest.log
cd /d "%~dp0.."
uv run python -u -m tests.roadmap_selftest projects\gpu_test > tests\roadmap_selftest.log 2>&1
echo exit %errorlevel% >> tests\roadmap_selftest.log
