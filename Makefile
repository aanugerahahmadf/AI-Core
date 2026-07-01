# ==============================================================================
# Wedding CBIR AI Core — Makefile
# ==============================================================================
# Usage:
#   make help          — tampilkan semua perintah
#   make install       — install dependencies
#   make sync          — rebuild index dari dataset.csv
#   make test          — jalankan semua tests
#   make lint          — jalankan pylint
#   make server        — jalankan Flask server
# ==============================================================================

# --- Config -------------------------------------------------------------------
PYTHON     := D:\laragon\bin\python\Python-3.14\python.exe
PIP        := $(PYTHON) -m pip
PYTEST     := $(PYTHON) -m pytest
PYLINT     := $(PYTHON) -m pylint
APP_URL    := http://127.0.0.1:8000
CSV_PATH   := data/dataset.csv

# --- Default target -----------------------------------------------------------
.DEFAULT_GOAL := help

.PHONY: help install install-dev sync rebuild test test-api test-cbir lint \
        server clean clean-cache clean-uploads prepare check

# ==============================================================================
# HELP
# ==============================================================================

help:
	@echo.
	@echo   Wedding CBIR AI Core
	@echo   =====================
	@echo.
	@echo   Setup:
	@echo     make install       Install semua dependencies
	@echo     make install-dev   Install + dev tools (pylint, pytest)
	@echo     make prepare       Buat folder data yang dibutuhkan
	@echo.
	@echo   Data:
	@echo     make sync          Rebuild index dari dataset.csv (+ sync ke CSV)
	@echo     make rebuild       Alias untuk sync
	@echo.
	@echo   Testing:
	@echo     make test          Jalankan semua tests
	@echo     make test-api      Jalankan tests Flask API saja
	@echo     make test-cbir     Jalankan tests CBIR core saja
	@echo.
	@echo   Quality:
	@echo     make lint          Jalankan pylint (target: 10.00/10)
	@echo     make check         lint + test sekaligus
	@echo.
	@echo   Server:
	@echo     make server        Jalankan Flask server (port 5000)
	@echo.
	@echo   Cleanup:
	@echo     make clean         Hapus __pycache__ dan .pyc files
	@echo     make clean-cache   Hapus cache pytest
	@echo     make clean-uploads Hapus file upload sementara
	@echo.

# ==============================================================================
# SETUP
# ==============================================================================

install:
	@echo [INSTALL] Installing dependencies...
	$(PIP) install flask flask-cors werkzeug python-dotenv
	$(PIP) install opencv-python Pillow numpy scipy requests tqdm
	$(PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	@echo [INSTALL] Done!

install-dev: install
	@echo [INSTALL-DEV] Installing dev tools...
	$(PIP) install pylint pytest
	@echo [INSTALL-DEV] Done!

prepare:
	@echo [PREPARE] Creating required directories...
	@if not exist data\inputs   mkdir data\inputs
	@if not exist data\features mkdir data\features
	@if not exist data\uploads  mkdir data\uploads
	@if not exist data\evaluation mkdir data\evaluation
	@if not exist reports\images  mkdir reports\images
	@echo [PREPARE] Done!

# ==============================================================================
# DATA — Rebuild Index
# ==============================================================================

sync:
	@echo [SYNC] Rebuilding CBIR index from $(CSV_PATH)...
	$(PYTHON) rebuild_index.py --csv $(CSV_PATH) --app-url $(APP_URL)

rebuild: sync

# ==============================================================================
# TESTING
# ==============================================================================

test:
	@echo [TEST] Running all tests...
	$(PYTEST) tests/ -v --tb=short

test-api:
	@echo [TEST-API] Running Flask API tests...
	$(PYTEST) tests/test_api.py -v --tb=short

test-cbir:
	@echo [TEST-CBIR] Running CBIR core tests...
	$(PYTEST) tests/test_cbir.py -v --tb=short

# ==============================================================================
# QUALITY
# ==============================================================================

lint:
	@echo [LINT] Running pylint...
	$(PYLINT) src/ app.py rebuild_index.py ai_sync.py \
		--rcfile=pylintrc --score=yes

check: lint test
	@echo [CHECK] All checks passed!

# ==============================================================================
# SERVER
# ==============================================================================

server:
	@echo [SERVER] Starting Flask server on port 5000...
	@echo [SERVER] Press Ctrl+C to stop
	$(PYTHON) app.py

# ==============================================================================
# CLEANUP
# ==============================================================================

clean:
	@echo [CLEAN] Removing __pycache__ and .pyc files...
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	@del /s /q *.pyc 2>nul || true
	@echo [CLEAN] Done!

clean-cache:
	@echo [CLEAN-CACHE] Removing pytest cache...
	@if exist .pytest_cache rd /s /q .pytest_cache
	@echo [CLEAN-CACHE] Done!

clean-uploads:
	@echo [CLEAN-UPLOADS] Removing temporary upload files...
	@del /q data\uploads\cbir-temp-*.jpg 2>nul || true
	@echo [CLEAN-UPLOADS] Done!
