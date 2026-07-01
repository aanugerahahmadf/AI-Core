@echo off
:: ==============================================================================
:: Wedding CBIR AI Core — make.bat (Windows equivalent of Makefile)
:: ==============================================================================
:: Usage:
::   make help          — tampilkan semua perintah
::   make install       — install dependencies
::   make sync          — rebuild index dari dataset.csv
::   make test          — jalankan semua tests
::   make lint          — jalankan pylint
::   make server        — jalankan Flask server
:: ==============================================================================

set PYTHON=D:\laragon\bin\python\python-3.13\python.exe
set PIP=%PYTHON% -m pip
set PYTEST=%PYTHON% -m pytest
set PYLINT=%PYTHON% -m pylint
set APP_URL=http://127.0.0.1:8000
set CSV_PATH=data\dataset.csv

if "%1"==""        goto help
if "%1"=="help"    goto help
if "%1"=="install"      goto install
if "%1"=="install-dev"  goto install_dev
if "%1"=="prepare"      goto prepare
if "%1"=="sync"         goto sync
if "%1"=="rebuild"      goto sync
if "%1"=="test"         goto test
if "%1"=="test-api"     goto test_api
if "%1"=="test-cbir"    goto test_cbir
if "%1"=="lint"         goto lint
if "%1"=="check"        goto check
if "%1"=="server"       goto server
if "%1"=="clean"        goto clean
if "%1"=="clean-cache"  goto clean_cache
if "%1"=="clean-uploads" goto clean_uploads

echo [ERROR] Unknown command: %1
goto help

:: ==============================================================================
:help
echo.
echo   Wedding CBIR AI Core
echo   =====================
echo.
echo   Setup:
echo     make install         Install semua dependencies
echo     make install-dev     Install + dev tools (pylint, pytest)
echo     make prepare         Buat folder data yang dibutuhkan
echo.
echo   Data:
echo     make sync            Rebuild index dari dataset.csv (+ sync ke CSV)
echo     make rebuild         Alias untuk sync
echo.
echo   Testing:
echo     make test            Jalankan semua tests
echo     make test-api        Jalankan tests Flask API saja
echo     make test-cbir       Jalankan tests CBIR core saja
echo.
echo   Quality:
echo     make lint            Jalankan pylint (target: 10.00/10)
echo     make check           lint + test sekaligus
echo.
echo   Server:
echo     make server          Jalankan Flask server (port 5000)
echo.
echo   Cleanup:
echo     make clean           Hapus __pycache__ dan .pyc files
echo     make clean-cache     Hapus cache pytest
echo     make clean-uploads   Hapus file upload sementara
echo.
goto end

:: ==============================================================================
:install
echo [INSTALL] Installing dependencies...
%PIP% install flask flask-cors werkzeug python-dotenv
%PIP% install opencv-python Pillow numpy scipy requests tqdm
%PIP% install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo [INSTALL] Done!
goto end

:install_dev
call %0 install
echo [INSTALL-DEV] Installing dev tools...
%PIP% install pylint pytest
echo [INSTALL-DEV] Done!
goto end

:prepare
echo [PREPARE] Creating required directories...
if not exist data\inputs    mkdir data\inputs
if not exist data\features  mkdir data\features
if not exist data\uploads   mkdir data\uploads
if not exist data\evaluation mkdir data\evaluation
if not exist reports\images  mkdir reports\images
echo [PREPARE] Done!
goto end

:: ==============================================================================
:sync
echo [SYNC] Rebuilding CBIR index from %CSV_PATH%...
%PYTHON% rebuild_index.py --csv %CSV_PATH% --app-url %APP_URL%
goto end

:: ==============================================================================
:test
echo [TEST] Running all tests...
%PYTEST% tests/ -v --tb=short
goto end

:test_api
echo [TEST-API] Running Flask API tests...
%PYTEST% tests/test_api.py -v --tb=short
goto end

:test_cbir
echo [TEST-CBIR] Running CBIR core tests...
%PYTEST% tests/test_cbir.py -v --tb=short
goto end

:: ==============================================================================
:lint
echo [LINT] Running pylint...
%PYLINT% src/ app.py rebuild_index.py ai_sync.py cbir_engine.py --rcfile=pylintrc --score=yes
goto end

:check
call %0 lint
if errorlevel 1 (
    echo [CHECK] Lint failed!
    exit /b 1
)
call %0 test
if errorlevel 1 (
    echo [CHECK] Tests failed!
    exit /b 1
)
echo [CHECK] All checks passed!
goto end

:: ==============================================================================
:server
echo [SERVER] Starting Flask server on port 5000...
echo [SERVER] Press Ctrl+C to stop
%PYTHON% app.py
goto end

:: ==============================================================================
:clean
echo [CLEAN] Removing __pycache__ and .pyc files...
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
del /s /q *.pyc 2>nul
echo [CLEAN] Done!
goto end

:clean_cache
echo [CLEAN-CACHE] Removing pytest cache...
if exist .pytest_cache rd /s /q .pytest_cache
echo [CLEAN-CACHE] Done!
goto end

:clean_uploads
echo [CLEAN-UPLOADS] Removing temporary upload files...
del /q data\uploads\cbir-temp-*.jpg 2>nul
echo [CLEAN-UPLOADS] Done!
goto end

:: ==============================================================================
:end
