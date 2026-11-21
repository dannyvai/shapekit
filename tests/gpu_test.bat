@echo off
REM One-time setup + GPU end-to-end test (uv sync, CUDA check, ALIKED + GPU depth maps on one clip). Log: tests\gpu_test.log
cd /d "%~dp0.."
set LOG=tests\gpu_test.log
echo === %date% %time% === > %LOG%
echo [1/4] uv sync >> %LOG%
uv sync >> %LOG% 2>&1
if errorlevel 1 (echo uv sync FAILED >> %LOG% & echo uv sync failed - see %LOG% & pause & exit /b 1)
echo [2/4] torch / CUDA check >> %LOG%
uv run python -c "import torch, lightglue, kornia; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print('arch list', torch.cuda.get_arch_list()); x=torch.randn(2048,2048,device='cuda'); print('matmul ok', float((x@x).sum())>-1e30)" >> %LOG% 2>&1
where ffmpeg >> %LOG% 2>&1
echo [3/4] ALIKED+LightGlue + GPU depth maps on one clip >> %LOG%
if exist projects\gpu_test rmdir /s /q projects\gpu_test
uv run python -m shapekit process projects\gpu_test --video videos\area_c\20261118_161654.mp4 --preset balanced --features aliked >> %LOG% 2>&1
echo [4/4] exit code %errorlevel% >> %LOG%
echo === done %date% %time% === >> %LOG%
echo Done. You can close this window.
pause
