@echo off
REM Retrieval benchmark on projects\area_d (SALAD / DINOv2 / VLAD / hybrid, verified with ALIKED+LightGlue). Log: benchmarks\retrieval_test.log
cd /d "%~dp0.."
uv run python -u -c "from tests.feature_selftest import retrieval_test; retrieval_test(r'projects\area_d')" > benchmarks\retrieval_test.log 2>&1
echo exit %errorlevel% >> benchmarks\retrieval_test.log
