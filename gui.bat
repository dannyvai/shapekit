@echo off
cd /d "%~dp0"
uv run python -m shapekit gui %* > gui.log 2>&1
if errorlevel 1 (type gui.log & pause)
