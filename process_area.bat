@echo off
REM Drag a folder of videos (or a single video) onto this file, or run:
REM   process_area.bat C:\workspace\photogrammetry\videos\area_c [fast|balanced|quality] [aliked|superpoint|disk|sift]
setlocal
cd /d "%~dp0"
set SRC=%~1
set NAME=%~n1
set PRESET=%~2
set FEAT=%~3
if "%PRESET%"=="" set PRESET=balanced
if "%FEAT%"=="" set FEAT=auto
if "%SRC%"=="" (
  echo Usage: process_area.bat ^<video folder or file^> [fast^|balanced^|quality] [aliked^|superpoint^|disk^|sift]
  pause
  exit /b 1
)
C:\workspace\photogrammetry\.venv\Scripts\python.exe -m shapekit process "projects\%NAME%" --video "%SRC%" --preset %PRESET% --features %FEAT%
if errorlevel 1 (echo Processing failed. & pause & exit /b 1)
start "" "projects\%NAME%\report.html"
pause
