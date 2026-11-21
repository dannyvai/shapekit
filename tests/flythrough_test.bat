@echo off
REM Renders all fly-through patterns (720p, 8 s) of projects\gpu_test into projects\gpu_test\exports. Log: tests\flythrough_test.log
cd /d "%~dp0.."
set LOG=tests\flythrough_test.log
echo === %date% %time% === > %LOG%
for %%p in (orbit helix flyover reveal topdown capture) do uv run python -u -m shapekit flythrough projects\gpu_test --pattern %%p --seconds 8 --size 720p >> %LOG% 2>&1
echo exit %errorlevel% >> %LOG%
