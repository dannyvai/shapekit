@echo off
REM Renders the README gallery (stills, photo vs model, fly-through MP4 + GIF, GUI screenshot) from projects\bariniki into docs\media. Log: docs\make_media.log
cd /d "%~dp0.."
uv run python -u docs\make_media.py projects\bariniki > docs\make_media.log 2>&1
echo exit %errorlevel% >> docs\make_media.log
