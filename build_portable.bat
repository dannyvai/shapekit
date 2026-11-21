@echo off
rem Builds the portable ShapeKit folder: dist\ShapeKit (ShapeKit.exe + shapekit-cli.exe + models + ffmpeg).
rem Takes 10-25 minutes. Log: build_portable.log
cd /d "%~dp0"
call :main > build_portable.log 2>&1
echo exit %errorlevel% >> build_portable.log
goto :eof
:main
echo [1/5] downloading model weights into models\ ...
uv run python -u -m shapekit.tools.prefetch_models models || exit /b 1
echo [2/5] PyInstaller build ...
if exist dist\ShapeKit rmdir /s /q dist\ShapeKit
uv run --with "pyinstaller>=6.11" --with "pyinstaller-hooks-contrib>=2025.1" pyinstaller --noconfirm --clean --distpath dist --workpath build packaging\shapekit.spec || exit /b 2
echo [3/5] adding models, ffmpeg, readme ...
uv run python -u packaging\finish_build.py || exit /b 3
echo [4/5] smoke test of the frozen app ...
set TORCH_HOME=
dist\ShapeKit\shapekit-cli.exe doctor projects\gpu_test || exit /b 4
echo [5/5] BUILD OK: dist\ShapeKit
exit /b 0
